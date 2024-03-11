"""
This file contains code to compute derived scalars related to attention heads, such as the norm of
the vector written by an attention head to the residual stream between a pair of tokens, or the
lower triangle of attention scores from later tokens to earlier tokens, flattened into a single
vector.
"""

from typing import Callable

import torch
import torch.nn

from neuron_explainer.activations.derived_scalars.autoencoder import (
    make_autoencoder_activation_fn_derivative,
    make_autoencoder_latent_grad_wrt_residual_input_scalar_source,
    make_autoencoder_latent_scalar_deriver_factory,
    make_autoencoder_pre_act_encoder_derivative,
)
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.direct_effects import (
    compute_attn_write_to_residual_direction_from_attn_weighted_values,
    convert_scalar_deriver_to_write_to_final_residual_grad,
)
from neuron_explainer.activations.derived_scalars.locations import ConstantLayerIndexer
from neuron_explainer.activations.derived_scalars.raw_activations import (
    get_scalar_sources_for_activation_location_types,
    make_scalar_deriver_factory_for_act_times_grad,
    make_scalar_deriver_factory_for_activation_location_type,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DerivedScalarSource,
    DstConfig,
    RawScalarSource,
    ScalarDeriver,
    ScalarSource,
)
from neuron_explainer.activations.derived_scalars.write_tensors import (
    get_attn_write_tensor_by_layer_index,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    LayerIndex,
    NodeType,
    PassType,
    WeightLocationType,
)
from neuron_explainer.models.model_context import ModelContext


def flatten_lower_triangle(t: torch.Tensor) -> torch.Tensor:
    """This turns the lower triangular entries of a tensor (that is, entries with
    i <= j for i and j indexing the last two dimensions) into a tensor flattened
    along the last two dimensions. This is used for turning attention-related activations
    which would normally be list[list[float]] for each token sequence, into a list[float],
    which is compatible with NeuronRecord's."""
    # Get the shape of the tensor
    shape = t.shape

    # Get the last dimension N (tensor should be ... x M x N)
    M, N = shape[-2:]

    # Get the indices of the lower triangular part
    row_indices, col_indices = torch.tril_indices(M, N)

    # Reshape the tensor so that all preceding dimensions are combined into one
    t_reshaped = t.reshape(-1, M, N)

    # Extract the lower triangular part and flatten it
    flattened_lower_triangle = t_reshaped[:, row_indices, col_indices]

    # Reshape back to separate out the original preceding dimensions
    new_shape = list(shape[:-2]) + [-1]
    flattened_lower_triangle = flattened_lower_triangle.reshape(new_shape)

    return flattened_lower_triangle


def _convert_rectangular_activations_to_flattened_lower_triangle(
    rectangular_activations: torch.Tensor,
) -> torch.Tensor:
    """
    This function takes a tensor with the first two dimensions indexed by rows i and columns j.
    It keeps the entries where i <= j and flattens the first 2 dimensions into a 1-d list.
    """

    assert rectangular_activations.ndim == 3, rectangular_activations.shape
    num_sequence_tokens, num_attended_to_sequence_tokens, nheads = rectangular_activations.shape
    rectangular_activations = torch.einsum(
        "fth->hft", rectangular_activations
    )  # (nheads, num_sequence_tokens, num_attended_to_sequence_tokens
    flattened_activations = flatten_lower_triangle(rectangular_activations)
    flattened_activations = torch.einsum(
        "hs->sh", flattened_activations
    )  # (num_token_pairs, nheads)
    min_num_tokens = min(num_sequence_tokens, num_attended_to_sequence_tokens)
    num_token_pairs_before_attended_to_sequence_is_saturated = (
        min_num_tokens * (min_num_tokens + 1) // 2
    )
    if num_sequence_tokens > num_attended_to_sequence_tokens:
        num_token_pairs_after_attended_to_sequence_is_saturated = (
            num_sequence_tokens - num_attended_to_sequence_tokens
        ) * num_attended_to_sequence_tokens
        num_token_pairs = (
            num_token_pairs_before_attended_to_sequence_is_saturated
            + num_token_pairs_after_attended_to_sequence_is_saturated
        )
    else:
        num_token_pairs = num_token_pairs_before_attended_to_sequence_is_saturated
    # if num_sequence_tokens == num_attended_to_sequence_tokens, then the i>=j entries form a lower triangle (first term)
    # if num_sequence_tokens > num_attended_to_sequence_tokens, then there is an additional rectangle of entries
    # below that lower triangle with i > j (second term)
    assert flattened_activations.shape == (
        num_token_pairs,
        nheads,
    ), f"{flattened_activations.shape=} != {(num_token_pairs, nheads)=}"
    return flattened_activations


def unflatten_lower_triangle(flattened: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Inverse of flatten_lower_triangle.

    This function reconstructs a tensor from its flattened lower triangular part.
    The upper triangular part of the reconstructed tensor is filled with zeros.
    """
    # Get the indices of the lower triangular part
    row_indices, col_indices = torch.tril_indices(M, N)

    # Create an empty tensor to fill in the original shape
    new_shape = list(flattened.shape[:-1]) + [M, N]
    reconstructed = torch.zeros(new_shape, dtype=flattened.dtype, device=flattened.device)

    # Fill in the lower triangular part of the reconstructed tensor
    reconstructed[..., row_indices, col_indices] = flattened

    return reconstructed


def unflatten_lower_triangle_and_sum_columns(
    flattened: torch.Tensor, M: int, N: int
) -> torch.Tensor:
    """Equivalent to unflatten_lower_triangle(...).sum(dim=-1), less memory, more time.

    This function calculates the sum (over the last dimension) of the lower triangular part of a tensor
    from its flattened representation without instantiating the full matrix.
    """
    # Get the indices of the lower triangular part
    row_indices, col_indices = torch.tril_indices(M, N)
    num_elements = row_indices.shape[0]
    assert flattened.shape[-1] == num_elements

    # Create an empty tensor to store the sum of each row
    new_shape = list(flattened.shape[:-1]) + [M]
    reconstructed_summed = torch.zeros(new_shape, dtype=flattened.dtype, device=flattened.device)

    # Sum the elements over columns
    for i in range(num_elements):
        reconstructed_summed[..., row_indices[i]] += flattened[..., i]

    return reconstructed_summed


def _compute_v_times_Wo_norm(
    attn_value: torch.Tensor,  # thd
    W_O: torch.Tensor,  # hdo
) -> torch.Tensor:
    """
    This function computes the norm of the product of attention values and the output weight matrix
    (W_O in the transformer-circuits convention, or c_proj in many internal settings).
    The attention values are represented by the tensor attn_value and the output weight matrix by W_O.
    The function returns a tensor representing the norm of the product.
    """

    num_attended_to_sequence_tokens, nheads, d_head = attn_value.shape
    assert W_O.shape[:2] == (nheads, d_head)  # third dim is d_model
    v_times_Wo = torch.einsum("thd,hdo->tho", attn_value, W_O)
    return torch.linalg.norm(v_times_Wo, dim=-1)


def _compute_attn_write_norm_from_attn_and_value(
    attn_post_softmax: torch.Tensor,  # fth
    attn_value: torch.Tensor,  # thd
    W_O: torch.Tensor,  # hdo
    pass_type: PassType,
) -> torch.Tensor:
    """Computes the norm of the write vector from each token to each other token,
    by multiplying two saved activations and a model weight.
    For the forward pass, this is attn_post_softmax * ||W_O @ V|| (elementwise multiplication).
    For the backward pass (the gradient with respect to the forward pass quantity),
    this is grad_attn_post_softmax / ||W_O @ V|| (elementwise division)"""
    # attn_post_softmax: (nheads, num_sequence_tokens, num_attended_to_sequence_tokens)
    # attn_value: (num_attended_to_sequence_tokens, nheads, d_head)
    # W_O: (nheads, d_head, d_model)
    # output: (num_sequence_tokens, num_attended_to_sequence_tokens, nheads)
    num_sequence_tokens, num_attended_to_sequence_tokens, nheads = attn_post_softmax.shape
    d_head = attn_value.shape[-1]
    assert attn_value.shape == (num_attended_to_sequence_tokens, nheads, d_head)
    v_times_Wo_norm = _compute_v_times_Wo_norm(attn_value, W_O)
    assert v_times_Wo_norm.shape == (num_attended_to_sequence_tokens, nheads)
    if pass_type == PassType.FORWARD:
        v_times_Wo_norm_factor = v_times_Wo_norm
    else:
        assert pass_type == PassType.BACKWARD
        v_times_Wo_norm_factor = 1 / v_times_Wo_norm
    output = torch.einsum("fth,th->fth", attn_post_softmax, v_times_Wo_norm_factor)
    assert output.shape == (num_sequence_tokens, num_attended_to_sequence_tokens, nheads)
    return output


def _get_attn_write_for_one_layer_index(
    model_context: ModelContext,
    layer_index: int,
) -> torch.Tensor:
    """Returns a dictionary mapping layer index to the write weight matrix for that layer."""
    return get_attn_write_tensor_by_layer_index(
        model_context=model_context,
        layer_indices=[layer_index],
    )[layer_index]


def _make_unflattened_attn_write_norm_tensor_calculate_derived_scalar_fn(
    model_context: ModelContext,
    layer_indices: list[int],
) -> Callable[[tuple[torch.Tensor, ...], LayerIndex, PassType], torch.Tensor]:
    """Returns a function that takes a tuple of tensors containing
    post softmax attention values and value vectors, and returns a tensor
    of the norm of the write vector from each token to each other token."""
    nheads = model_context.n_attention_heads
    d_model = model_context.n_residual_stream_channels
    assert all(isinstance(layer_index, int) for layer_index in layer_indices)

    W_O_by_layer_index = get_attn_write_tensor_by_layer_index(
        model_context=model_context,
        layer_indices=layer_indices,
    )

    def _attn_write_norm_tensor_calculate_derived_scalar_fn(
        raw_activation_data_tuple: tuple[torch.Tensor, ...],
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        """The returned processing hook, to be passed as an argument to the ScalarDeriver."""

        if len(raw_activation_data_tuple) == 2:
            attn_post_softmax_data, attn_value_data = raw_activation_data_tuple
            assert (
                pass_type == PassType.FORWARD
            ), f"Need gradient of attn_post_softmax for backward pass, got {raw_activation_data_tuple}"
        else:
            assert len(raw_activation_data_tuple) == 3
            # first position is forward pass attn_post_softmax,
            # second is backward pass attn_post_softmax,
            # third is attn_value
            if pass_type == PassType.FORWARD:
                attn_post_softmax_data, _, attn_value_data = raw_activation_data_tuple
            else:
                assert pass_type == PassType.BACKWARD
                _, attn_post_softmax_data, attn_value_data = raw_activation_data_tuple

        assert layer_index in W_O_by_layer_index

        return _compute_attn_write_norm_from_attn_and_value(
            attn_post_softmax=attn_post_softmax_data,
            attn_value=attn_value_data,
            W_O=W_O_by_layer_index[layer_index],
            pass_type=pass_type,
        )

    return _attn_write_norm_tensor_calculate_derived_scalar_fn


def make_unflattened_attn_write_norm_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a ScalarDeriver object for the attention write norm, between any
    two tokens. The actual attention write vector is the vector sum of many attention
    write vectors. Note that the norm of this vector sum will be less that the sum of the
    attention write norms computed here."""
    model_context = dst_config.get_model_context()
    layer_indices = dst_config.layer_indices or list(range(model_context.n_layers))

    if dst_config.derive_gradients:
        sub_scalar_sources: tuple[ScalarSource, ...] = (
            RawScalarSource(
                activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
                pass_type=PassType.FORWARD,
            ),
            RawScalarSource(
                activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
                pass_type=PassType.BACKWARD,
            ),
            RawScalarSource(
                activation_location_type=ActivationLocationType.ATTN_VALUE,
                pass_type=PassType.FORWARD,
            ),
        )
    else:
        sub_scalar_sources = (
            RawScalarSource(
                activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
                pass_type=PassType.FORWARD,
            ),
            RawScalarSource(
                activation_location_type=ActivationLocationType.ATTN_VALUE,
                pass_type=PassType.FORWARD,
            ),
        )

    return ScalarDeriver(
        dst=DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM,
        dst_config=dst_config,
        sub_scalar_sources=sub_scalar_sources,
        tensor_calculate_derived_scalar_fn=_make_unflattened_attn_write_norm_tensor_calculate_derived_scalar_fn(
            model_context=model_context,
            layer_indices=layer_indices,
        ),
    )


def make_attn_write_norm_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = make_unflattened_attn_write_norm_scalar_deriver(dst_config)
    return scalar_deriver.apply_transform_fn_to_output(
        _convert_rectangular_activations_to_flattened_lower_triangle,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.ATTN_WRITE_NORM,
    )


def make_flattened_attn_post_softmax_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_QK_PROBS
    )(dst_config)
    return scalar_deriver.apply_transform_fn_to_output(
        _convert_rectangular_activations_to_flattened_lower_triangle,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.FLATTENED_ATTN_POST_SOFTMAX,
    )


def make_flattened_attn_post_softmax_act_times_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = make_scalar_deriver_factory_for_act_times_grad(
        ActivationLocationType.ATTN_QK_PROBS,
        DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD,
    )(dst_config)
    return scalar_deriver.apply_transform_fn_to_output(
        _convert_rectangular_activations_to_flattened_lower_triangle,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.ATTN_ACT_TIMES_GRAD,
    )


def make_attn_act_times_grad_per_sequence_token_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    assert not dst_config.derive_gradients, "Gradients not defined for act times grad"
    scalar_deriver = make_scalar_deriver_factory_for_act_times_grad(
        ActivationLocationType.ATTN_QK_PROBS,
        DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD,
    )(dst_config)
    return scalar_deriver.apply_transform_fn_to_output(
        lambda activations: activations.sum(dim=1),
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.ATTN_ACT_TIMES_GRAD_PER_SEQUENCE_TOKEN,
    )


def make_attn_write_norm_per_sequence_token_tensor_calculate_derived_scalar_fn(
    model_context: ModelContext,
    layer_indices: list[int],
) -> Callable[[tuple[torch.Tensor, ...], LayerIndex, PassType], torch.Tensor]:
    W_O_by_layer_index = get_attn_write_tensor_by_layer_index(
        model_context=model_context,
        layer_indices=layer_indices,
    )

    def attn_write_norm_per_sequence_token_tensor_calculate_derived_scalar_fn(
        raw_activation_data_tuple: tuple[torch.Tensor, ...],
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert layer_index in W_O_by_layer_index
        assert len(raw_activation_data_tuple) == 1
        attn_weighted_sum_of_values = raw_activation_data_tuple[0]
        attn_write_norm_per_sequence_token = _compute_v_times_Wo_norm(
            attn_value=attn_weighted_sum_of_values,
            W_O=W_O_by_layer_index[layer_index],
        )
        return attn_write_norm_per_sequence_token

    return attn_write_norm_per_sequence_token_tensor_calculate_derived_scalar_fn


def make_attn_write_norm_per_sequence_token_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    activation_location_type = ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES
    sub_scalar_sources = get_scalar_sources_for_activation_location_types(
        activation_location_type, dst_config.derive_gradients
    )
    model_context = dst_config.get_model_context()
    layer_indices = dst_config.layer_indices or list(range(model_context.n_layers))
    attn_write_norm_per_sequence_token_tensor_calculate_derived_scalar_fn = (
        make_attn_write_norm_per_sequence_token_tensor_calculate_derived_scalar_fn(
            model_context=model_context,
            layer_indices=layer_indices,
        )
    )
    return ScalarDeriver(
        dst=DerivedScalarType.ATTN_WRITE_NORM_PER_SEQUENCE_TOKEN,
        dst_config=dst_config,
        sub_scalar_sources=sub_scalar_sources,
        tensor_calculate_derived_scalar_fn=attn_write_norm_per_sequence_token_tensor_calculate_derived_scalar_fn,
    )


def convert_attn_weighted_value_scalar_deriver_to_write_to_residual_direction_in_same_layer(
    attn_weighted_value_scalar_deriver: ScalarDeriver,
    direction_scalar_source: ScalarSource,
    output_dst: DerivedScalarType,
) -> ScalarDeriver:
    model_context = attn_weighted_value_scalar_deriver.dst_config.get_model_context()
    dst_config = attn_weighted_value_scalar_deriver.dst_config
    direction_layer_index = direction_scalar_source.layer_index

    assert dst_config.layer_indices is not None
    assert len(dst_config.layer_indices) == 1
    assert dst_config.layer_indices[0] == direction_layer_index

    W_O = _get_attn_write_for_one_layer_index(
        model_context=model_context,
        layer_index=direction_layer_index,
    )

    def attn_write_to_residual_direction_tensor_calculate_derived_scalar_fn(
        attn_weighted_values: torch.Tensor,
        residual_grad: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert layer_index == direction_layer_index
        assert (
            pass_type == PassType.FORWARD
        ), "write to final residual grad only defined for forward pass"
        return compute_attn_write_to_residual_direction_from_attn_weighted_values(
            attn_weighted_values=attn_weighted_values,
            residual_direction=residual_grad,
            W_O=W_O,
            pass_type=pass_type,
        )

    return attn_weighted_value_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
        layerwise_transform_fn=attn_write_to_residual_direction_tensor_calculate_derived_scalar_fn,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=output_dst,
        other_scalar_source=direction_scalar_source,
    )


def make_attn_write_to_final_residual_grad_per_sequence_token_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    attn_weighted_value_scalar_deriver = make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES,
    )(dst_config)
    return convert_scalar_deriver_to_write_to_final_residual_grad(
        scalar_deriver=attn_weighted_value_scalar_deriver,
        output_dst=DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD_PER_SEQUENCE_TOKEN,
        use_existing_backward_pass_for_final_residual_grad=True,
    )


def _check_attn_post_softmax_shape(attn_post_softmax: torch.Tensor) -> None:
    num_sequence_tokens, num_attended_to_sequence_tokens, nheads = attn_post_softmax.shape
    assert num_attended_to_sequence_tokens == num_sequence_tokens


def _compute_attn_write_from_attn_and_value(
    attn_post_softmax: torch.Tensor,  # fth
    attn_value: torch.Tensor,  # thd
    W_O: torch.Tensor,  # hdo (heads, d_head, d_model)
    pass_type: PassType,
) -> torch.Tensor:
    assert (
        pass_type == PassType.FORWARD
    ), "only forward pass implemented for now for attn write projection from value"
    _check_attn_post_softmax_shape(attn_post_softmax)
    num_attended_to_sequence_tokens, n_heads, d_head = attn_value.shape
    assert attn_post_softmax.shape[1] == num_attended_to_sequence_tokens
    assert attn_post_softmax.shape[2] == n_heads
    num_sequence_tokens, num_attended_to_sequence_tokens, n_heads = attn_post_softmax.shape
    assert W_O.shape[:2] == (n_heads, d_head)
    d_model = W_O.shape[2]
    attn_value_times_Wo = torch.einsum("thd,hdv->thv", attn_value, W_O)
    attn_weighted_v_times_Wo = torch.einsum("fth,thv->fthv", attn_post_softmax, attn_value_times_Wo)
    assert attn_weighted_v_times_Wo.shape == (
        num_sequence_tokens,
        num_attended_to_sequence_tokens,
        n_heads,
        d_model,
    )
    return attn_weighted_v_times_Wo


def make_unflattened_attn_write_to_final_residual_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = make_attn_weighted_value_scalar_deriver(
        dst_config=dst_config,
    )
    return convert_scalar_deriver_to_write_to_final_residual_grad(
        scalar_deriver=scalar_deriver,
        output_dst=DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
        use_existing_backward_pass_for_final_residual_grad=True,
    )


def make_unflattened_attn_write_to_final_activation_residual_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    attn_weighted_value_scalar_deriver = make_attn_weighted_value_scalar_deriver(
        dst_config=dst_config,
    )
    return convert_scalar_deriver_to_write_to_final_residual_grad(
        scalar_deriver=attn_weighted_value_scalar_deriver,
        output_dst=DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
        use_existing_backward_pass_for_final_residual_grad=False,
    )


def make_flattened_attn_write_to_final_residual_grad_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    scalar_deriver = make_unflattened_attn_write_to_final_residual_grad_scalar_deriver(dst_config)
    return scalar_deriver.apply_transform_fn_to_output(
        _convert_rectangular_activations_to_flattened_lower_triangle,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
    )


def make_attn_write_tensor_calculate_derived_scalar_fn(
    model_context: ModelContext,
    layer_indices: list[int],
) -> Callable[[tuple[torch.Tensor, ...], LayerIndex, PassType], torch.Tensor]:
    W_O_by_layer_index = get_attn_write_tensor_by_layer_index(
        model_context=model_context,
        layer_indices=layer_indices,
    )

    def attn_write_tensor_calculate_derived_scalar_fn(
        raw_activation_data_tuple: tuple[torch.Tensor, ...],
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert layer_index in W_O_by_layer_index
        assert len(raw_activation_data_tuple) == 2
        assert (
            pass_type == PassType.FORWARD
        ), "write to residual stream only implemented for forward pass"
        attn_post_softmax, attn_value = raw_activation_data_tuple
        attn_write_per_sequence_token = _compute_attn_write_from_attn_and_value(
            attn_post_softmax=attn_post_softmax,
            attn_value=attn_value,
            W_O=W_O_by_layer_index[layer_index],
            pass_type=pass_type,
        )
        return attn_write_per_sequence_token

    return attn_write_tensor_calculate_derived_scalar_fn


def make_attn_write_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a ScalarDeriver object for the attention write, between any two tokens,
    and for each head."""
    model_context = dst_config.get_model_context()
    layer_indices = dst_config.layer_indices or list(range(model_context.n_layers))

    assert not dst_config.derive_gradients
    sub_scalar_sources = (
        RawScalarSource(
            activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
            pass_type=PassType.FORWARD,
        ),
        RawScalarSource(
            activation_location_type=ActivationLocationType.ATTN_VALUE, pass_type=PassType.FORWARD
        ),
    )

    return ScalarDeriver(
        dst=DerivedScalarType.ATTN_WRITE,
        dst_config=dst_config,
        sub_scalar_sources=sub_scalar_sources,
        tensor_calculate_derived_scalar_fn=make_attn_write_tensor_calculate_derived_scalar_fn(
            model_context=model_context,
            layer_indices=layer_indices,
        ),
    )


def make_attn_write_sum_heads_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a ScalarDeriver object for the attention write, between any two tokens,
    summed over all heads."""
    scalar_deriver = make_attn_write_scalar_deriver(dst_config)

    def _sum_heads(attn_write: torch.Tensor) -> torch.Tensor:
        n_tokens_written_to, n_tokens_attended_to, n_heads, d_model = attn_write.shape
        return attn_write.sum(dim=2)

    return scalar_deriver.apply_transform_fn_to_output(
        _sum_heads,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.ATTN_WRITE_SUM_HEADS,
    )


def make_attn_weighted_value_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the attention value weighted by the post-softmax
    attention between each pair of tokens. Output shape: (n_tokens, n_attended_to_tokens, n_heads, v_channels)
    """

    def attn_weighted_value_tensor_calculate_derived_scalar_fn(
        raw_activation_data_tuple: tuple[torch.Tensor, ...],
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert len(raw_activation_data_tuple) == 2, ([t.shape for t in raw_activation_data_tuple],)
        assert pass_type == PassType.FORWARD, "weighted value only implemented for forward pass"
        attn_post_softmax, attn_value = raw_activation_data_tuple
        _check_attn_post_softmax_shape(attn_post_softmax)
        attn_weighted_value = torch.einsum("fth,thd->fthd", attn_post_softmax, attn_value)
        return attn_weighted_value

    return ScalarDeriver(
        dst=DerivedScalarType.ATTN_WEIGHTED_VALUE,
        dst_config=dst_config,
        sub_scalar_sources=(
            RawScalarSource(
                activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
                pass_type=PassType.FORWARD,
            ),
            RawScalarSource(
                activation_location_type=ActivationLocationType.ATTN_VALUE,
                pass_type=PassType.FORWARD,
            ),
        ),
        tensor_calculate_derived_scalar_fn=attn_weighted_value_tensor_calculate_derived_scalar_fn,
    )


def make_attn_write_to_latent_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns the attention value weighted by the post-softmax attention between each pair of tokens,
    and projected to the gradient of autoencoder latent wrt input.

    Output has shape (n_tokens, n_attended_to_tokens, n_heads).
    """

    attn_weighted_value_scalar_deriver = make_attn_weighted_value_scalar_deriver(
        dst_config
    )  # derive scalar of shape (n_tokens, n_attended_to_tokens, n_heads, v_channels)

    direction_scalar_source = make_autoencoder_latent_grad_wrt_residual_input_scalar_source(
        dst_config, NodeType.ATTENTION_AUTOENCODER_LATENT
    )  # derive scalar of shape (res_channels) = (n_heads * v_channels)
    return convert_attn_weighted_value_scalar_deriver_to_write_to_residual_direction_in_same_layer(
        attn_weighted_value_scalar_deriver=attn_weighted_value_scalar_deriver,
        direction_scalar_source=direction_scalar_source,
        output_dst=DerivedScalarType.ATTN_WRITE_TO_LATENT,
    )  # derive scalar of shape (n_tokens, n_attended_to_tokens, n_heads)


def make_attn_write_to_latent_summed_over_heads_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns the attention value weighted by the post-softmax attention between each pair of tokens,
     summed over the heads, and projected to the gradient of autoencoder latent wrt input.

    Output has shape (n_tokens, n_attended_to_tokens).
    """

    attn_write_to_latent_scalar_deriver = make_attn_write_to_latent_scalar_deriver(
        dst_config
    )  # derive scalar of shape (n_tokens, n_attended_to_tokens, n_heads)

    return attn_write_to_latent_scalar_deriver.apply_transform_fn_to_output(
        lambda activations: activations.sum(dim=-1)[..., None],
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS,
    )  # derive scalar of shape (n_tokens, n_attended_to_tokens, 1)


def make_flattened_attn_write_to_latent_summed_over_heads_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns the attention value weighted by the post-softmax attention between each pair of tokens,
    summed over the heads, and projected to the gradient of autoencoder latent wrt input,\
    flattening the lower triangular part of the output.

    Output has shape (n_token_pairs).
    """
    attn_write_to_latent_scalar_deriver = make_attn_write_to_latent_scalar_deriver(
        dst_config
    )  # derive scalar of shape (n_tokens, n_attended_to_tokens, n_heads)

    return attn_write_to_latent_scalar_deriver.apply_transform_fn_to_output(
        lambda activations: _convert_rectangular_activations_to_flattened_lower_triangle(
            activations.sum(dim=-1)[..., None]
        ),
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS,
    )  # derive scalar of shape (n_token_pairs, 1)


def make_flattened_attn_write_to_latent_summed_over_heads_batched_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Same as make_flattened_attn_write_to_latent_summed_over_heads_scalar_deriver, but over
    all latents at the same time.
    Assumes that the autoencoder has a single encoder layer and a ReLU, to compute the
    gradient (attribution) without backprop.
    Output has shape (n_token_pairs, n_latents).
    """
    autoencoder_node_type = NodeType.ATTENTION_AUTOENCODER_LATENT
    attn_write_sum_heads_scalar_deriver = make_attn_write_sum_heads_scalar_deriver(
        dst_config
    )  # derive scalar of shape (n_tokens, n_attended_to_tokens, d_model)
    autoencoder_scalar_deriver = make_autoencoder_latent_scalar_deriver_factory(
        autoencoder_node_type
    )(
        dst_config
    )  # derive scalar of shape (n_tokens, n_latents)

    assert dst_config.layer_indices is not None
    assert len(dst_config.layer_indices) == 1
    layer_index = dst_config.layer_indices[0]

    autoencoder_context = dst_config.get_autoencoder_context(autoencoder_node_type)
    assert autoencoder_context is not None
    pre_act_encoder_derivative = make_autoencoder_pre_act_encoder_derivative(
        autoencoder_context, layer_index
    )
    activation_fn_derivative = make_autoencoder_activation_fn_derivative(
        autoencoder_context, layer_index
    )

    def attn_write_to_latents_tensor_calculate_derived_scalar_fn(
        attn_write_sum_heads: torch.Tensor,
        autoencoder_latents: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert layer_index == layer_index
        assert pass_type == PassType.FORWARD, "write to latents only defined for forward pass"
        n_tokens, n_tokens_attended_to, d_model = attn_write_sum_heads.shape
        assert autoencoder_latents.shape[0] == n_tokens
        n_tokens, n_latents = autoencoder_latents.shape
        # from (n_tokens, n_tokens_attended_to, d_model) to (n_tokens, n_tokens_attended_to, n_latents)
        projection = pre_act_encoder_derivative(attn_write_sum_heads)

        d_latent_d_pre_act = activation_fn_derivative(autoencoder_latents)
        direct_write_to_latents = torch.einsum("tul,tl->tul", projection, d_latent_d_pre_act)

        flattened_direct_write_to_latents = (
            _convert_rectangular_activations_to_flattened_lower_triangle(direct_write_to_latents)
        )
        return flattened_direct_write_to_latents

    new_scalar_deriver = (
        attn_write_sum_heads_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
            layerwise_transform_fn=attn_write_to_latents_tensor_calculate_derived_scalar_fn,
            pass_type_to_transform=PassType.FORWARD,
            output_dst=DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED,
            other_scalar_source=DerivedScalarSource(
                scalar_deriver=autoencoder_scalar_deriver,
                pass_type=PassType.FORWARD,
                layer_indexer=ConstantLayerIndexer(layer_index),
            ),
        )
    )
    return new_scalar_deriver  # derive scalar of shape (n_token_pairs, n_latents)


def make_attn_write_to_latent_per_sequence_token_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns the attention value weighted by the post-softmax attention between each pair of tokens,
     summed over the attended-to tokens, and projected to the gradient of autoencoder latent wrt input.

    Output has shape (n_tokens, n_heads).
    """
    attn_weighted_sum_of_value_scalar_deriver = (
        make_scalar_deriver_factory_for_activation_location_type(
            ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES
        )(dst_config)
    )  # derive scalar of shape (n_tokens, n_heads, v_channels)
    direction_scalar_source = make_autoencoder_latent_grad_wrt_residual_input_scalar_source(
        dst_config
    )  # derive scalar of shape (res_channels) = (n_heads * v_channels)
    return convert_attn_weighted_value_scalar_deriver_to_write_to_residual_direction_in_same_layer(
        attn_weighted_value_scalar_deriver=attn_weighted_sum_of_value_scalar_deriver,
        direction_scalar_source=direction_scalar_source,
        output_dst=DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN,
    )  # derive scalar of shape (n_tokens, n_heads)


def make_attn_write_to_latent_per_sequence_token_batched_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Same as make_attn_write_to_latent_per_sequence_token_scalar_deriver, but over
    all latents at the same time.

    Assumes that the autoencoder has a single encoder layer and a ReLU, to compute the
    gradient (attribution) without backprop.

    Output has shape (n_tokens, n_heads, n_latents).
    """
    autoencoder_node_type = NodeType.ATTENTION_AUTOENCODER_LATENT
    attn_weighted_sum_of_value_scalar_deriver = (
        make_scalar_deriver_factory_for_activation_location_type(
            ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES
        )(dst_config)
    )  # derive scalar of shape (n_tokens, n_heads, v_channels)

    autoencoder_scalar_deriver = make_autoencoder_latent_scalar_deriver_factory(
        autoencoder_node_type
    )(
        dst_config
    )  # derive scalar of shape (n_tokens, n_latents)

    assert dst_config.layer_indices is not None
    assert len(dst_config.layer_indices) == 1
    layer_index = dst_config.layer_indices[0]

    autoencoder_context = dst_config.get_autoencoder_context(autoencoder_node_type)
    assert autoencoder_context is not None
    pre_act_encoder_derivative = make_autoencoder_pre_act_encoder_derivative(
        autoencoder_context, layer_index
    )
    activation_fn_derivative = make_autoencoder_activation_fn_derivative(
        autoencoder_context, layer_index
    )

    model_context = dst_config.get_model_context()
    W_O = model_context.get_weight(
        location_type=WeightLocationType.ATTN_TO_RESIDUAL,
        layer=layer_index,
        device=model_context.device,
    )  # shape (n_heads, d_head, d_model)

    def attn_write_to_latents_tensor_calculate_derived_scalar_fn(
        attn_weighted_sum_of_value: torch.Tensor,
        autoencoder_latents: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert layer_index == layer_index
        assert pass_type == PassType.FORWARD, "write to latents only defined for forward pass"
        n_tokens, n_heads, d_head = attn_weighted_sum_of_value.shape
        assert autoencoder_latents.shape[0] == n_tokens
        n_tokens, n_latents = autoencoder_latents.shape
        assert W_O.shape[:2] == (n_heads, d_head)
        n_heads, d_head, d_model = W_O.shape

        # from (n_heads, d_head, d_model) to (n_heads, d_head, n_latents)
        Wo_encoder = pre_act_encoder_derivative(W_O)
        Wo_encoder = Wo_encoder.to(attn_weighted_sum_of_value.dtype)

        projection = torch.einsum("thd,hdl->thl", attn_weighted_sum_of_value, Wo_encoder)

        d_latent_d_pre_act = activation_fn_derivative(autoencoder_latents)
        direct_write_to_latents = torch.einsum("thl,tl->thl", projection, d_latent_d_pre_act)
        return direct_write_to_latents

    new_scalar_deriver = attn_weighted_sum_of_value_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
        layerwise_transform_fn=attn_write_to_latents_tensor_calculate_derived_scalar_fn,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED,
        other_scalar_source=DerivedScalarSource(
            scalar_deriver=autoencoder_scalar_deriver,
            pass_type=PassType.FORWARD,
            layer_indexer=ConstantLayerIndexer(layer_index),
        ),
    )
    return new_scalar_deriver  # derive scalar of shape (n_token_pairs, n_heads, n_latents)


def make_reshape_fn(dst: DerivedScalarType) -> Callable:
    """Create a reshape function to apply to the output tensors."""

    output_dim = dst.shape_spec_per_token_sequence
    error_msg = f"Unexpected output_dim: {output_dim}. Please add a reshape function."

    if len(output_dim) == 2:
        # Regular activations are already 2d and don't need to be reshaped.
        assert output_dim[0] == Dimension.SEQUENCE_TOKENS
        assert output_dim[1].is_model_intrinsic
        reshape_fn = lambda x: x

    elif len(output_dim) == 3:
        assert output_dim[0] == Dimension.SEQUENCE_TOKENS
        assert output_dim[2].is_model_intrinsic
        if output_dim[1] == Dimension.ATTENDED_TO_SEQUENCE_TOKENS:
            # E.g. attention activations that are indexed both by current-token and token-attended-to.
            # Here, we move the two indexing dimensions to the end, we extract the lower triangle indices,
            # we flatten the lower triangle indices into a single dimension, and we move that dimension to the front.
            reshape_fn = lambda x: flatten_lower_triangle(x.permute(2, 0, 1)).permute(1, 0)
        elif output_dim[1] == Dimension.ATTN_HEADS:
            # E.g. attention activations that are split by attention heads.
            # Here, we merge the two model dimensions into one.
            reshape_fn = lambda x: x.reshape(x.shape[0], -1)
        else:
            raise NotImplementedError(error_msg)

    elif len(output_dim) == 4:
        assert output_dim[0] == Dimension.SEQUENCE_TOKENS
        assert output_dim[3].is_model_intrinsic
        if (
            output_dim[1] == Dimension.ATTENDED_TO_SEQUENCE_TOKENS
            and output_dim[2] == Dimension.ATTN_HEADS
        ):
            # Here, we move the two indexing dimensions to the end, we extract the lower triangle indices,
            # we flatten the lower triangle indices into a single dimension, and we move that dimension to the front.
            # Then we merge the merged input dimension with the attention heads dimension.
            reshape_fn = (
                lambda x: flatten_lower_triangle(x.permute(2, 3, 0, 1))
                .permute(2, 0, 1)
                .reshape(-1, x.shape[3])
            )
        else:
            raise NotImplementedError(error_msg)

    else:
        raise NotImplementedError(error_msg)
    return reshape_fn
