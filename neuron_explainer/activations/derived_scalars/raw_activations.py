"""
This file contains code to make scalar derivers for scalar types that are 1:1 with an
ActivationLocationType.
"""

from functools import partial
from typing import Callable

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.locations import (
    IdentityLayerIndexer,
    LayerIndexer,
    NoLayersLayerIndexer,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DstConfig,
    PassType,
    RawScalarSource,
    ScalarDeriver,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    LayerIndex,
    NodeType,
)


def get_scalar_sources_for_activation_location_types(
    activation_location_type: ActivationLocationType,
    derive_gradients: bool,
) -> tuple[RawScalarSource, ...]:
    if activation_location_type.has_no_layers:
        layer_indexer: LayerIndexer = NoLayersLayerIndexer()
    else:
        layer_indexer = IdentityLayerIndexer()
    if derive_gradients:
        return (
            RawScalarSource(
                activation_location_type=activation_location_type,
                pass_type=PassType.FORWARD,
                layer_indexer=layer_indexer,
            ),
            RawScalarSource(
                activation_location_type=activation_location_type,
                pass_type=PassType.BACKWARD,
                layer_indexer=layer_indexer,
            ),
        )
    else:
        return (
            RawScalarSource(
                activation_location_type=activation_location_type,
                pass_type=PassType.FORWARD,
                layer_indexer=layer_indexer,
            ),
        )


def no_op_tensor_calculate_derived_scalar_fn(
    raw_activation_data_tuple: tuple[torch.Tensor, ...],
    layer_index: LayerIndex,
    pass_type: PassType,
) -> torch.Tensor:
    """
    This either:
    converts a length 1 tuple of tensors into
    a single tensor; pass_type is asserted to be PassType.FORWARD
    or
    converts a length 2 tuple of tensors, one for the forward pass and one for the backward pass,
    into the appropriate one of those two objects, depending on the pass_type argument.
    """
    if len(raw_activation_data_tuple) == 1:
        # in this case, only the activations at the relevant ActivationLocationType have been loaded from disk
        assert pass_type == PassType.FORWARD
        raw_activation_data = raw_activation_data_tuple[0]
        return raw_activation_data
    elif len(raw_activation_data_tuple) == 2:
        # in this case, both the activations and gradients at the relevant ActivationLocationType have been loaded from disk
        raw_activation_data, raw_gradient_data = raw_activation_data_tuple
        if pass_type == PassType.FORWARD:
            return raw_activation_data
        elif pass_type == PassType.BACKWARD:
            return raw_gradient_data
        else:
            raise ValueError(f"Unknown {pass_type=}")
    else:
        raise ValueError(f"Unknown {raw_activation_data_tuple=}")


def make_scalar_deriver_factory_for_activation_location_type(
    activation_location_type: ActivationLocationType,
) -> Callable[[DstConfig], ScalarDeriver]:
    """
    This is for DerivedScalarType's 1:1 with a ActivationLocationType, which can be generated from
    just the ActivationLocationType and no additional information.
    """

    def make_scalar_deriver_fn(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        sub_scalar_sources = get_scalar_sources_for_activation_location_types(
            activation_location_type, dst_config.derive_gradients
        )

        return ScalarDeriver(
            dst=DerivedScalarType.from_activation_location_type(activation_location_type),
            dst_config=dst_config,
            sub_scalar_sources=sub_scalar_sources,
            tensor_calculate_derived_scalar_fn=no_op_tensor_calculate_derived_scalar_fn,
        )

    return make_scalar_deriver_fn


def make_scalar_deriver_factory_for_act_times_grad(
    activation_location_type: ActivationLocationType,
    dst: DerivedScalarType,
) -> Callable[[DstConfig], ScalarDeriver]:
    def make_act_times_grad_scalar_deriver(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        assert not dst_config.derive_gradients, "Gradients not defined for act times grad"
        if activation_location_type.has_no_layers:
            layer_indexer: LayerIndexer = NoLayersLayerIndexer()
        else:
            layer_indexer = IdentityLayerIndexer()
        sub_scalar_sources = (
            RawScalarSource(
                activation_location_type=activation_location_type,
                pass_type=PassType.FORWARD,
                layer_indexer=layer_indexer,
            ),  # activations
            RawScalarSource(
                activation_location_type=activation_location_type,
                pass_type=PassType.BACKWARD,
                layer_indexer=layer_indexer,
            ),  # gradients
        )

        def _act_times_grad_tensor_calculate_derived_scalar_fn(
            raw_activation_data_tuple: tuple[torch.Tensor, ...],
            layer_index: LayerIndex,
            pass_type: PassType,
        ) -> torch.Tensor:
            assert pass_type == PassType.FORWARD, "Backward pass not defined for act times grad"
            assert len(raw_activation_data_tuple) == 2
            raw_activation_data, raw_gradient_data = raw_activation_data_tuple
            return raw_activation_data * raw_gradient_data

        return ScalarDeriver(
            dst=dst,
            dst_config=dst_config,
            sub_scalar_sources=sub_scalar_sources,
            tensor_calculate_derived_scalar_fn=_act_times_grad_tensor_calculate_derived_scalar_fn,
        )

    return make_act_times_grad_scalar_deriver


def check_write_tensor_device_matches(
    scalar_deriver: ScalarDeriver,
    write_tensor_by_layer_index: dict[LayerIndex, torch.Tensor] | dict[int, torch.Tensor],
) -> None:
    write_matrix_device = next(iter(write_tensor_by_layer_index.values())).device
    assert scalar_deriver.device_for_raw_activations == write_matrix_device, (
        scalar_deriver.dst,
        scalar_deriver.device_for_raw_activations,
        write_matrix_device,
    )


def convert_scalar_deriver_to_write_norm(
    scalar_deriver: ScalarDeriver,
    write_tensor_by_layer_index: dict[LayerIndex, torch.Tensor] | dict[int, torch.Tensor],
    output_dst: DerivedScalarType,
) -> ScalarDeriver:
    """
    Converts a scalar deriver for a scalar type that is 1:1 with an ActivationLocationType to a
    scalar deriver for the write norm for each neuron at each token.
    """

    check_write_tensor_device_matches(
        scalar_deriver,
        write_tensor_by_layer_index,
    )

    write_norm_by_layer_index = {
        layer_index_: write_tensor_by_layer_index[layer_index_].norm(dim=-1)  # type: ignore
        for layer_index_ in write_tensor_by_layer_index.keys()
    }

    def multiply_by_write_norm(
        activations: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD, "Backward pass not defined for write norm"
        assert (
            layer_index in write_tensor_by_layer_index
        ), f"{layer_index=} not in {write_tensor_by_layer_index.keys()=} for {output_dst=}"
        return activations * write_norm_by_layer_index[layer_index]

    return scalar_deriver.apply_layerwise_transform_fn_to_output(
        multiply_by_write_norm,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=output_dst,
    )


def convert_scalar_deriver_to_write(
    scalar_deriver: ScalarDeriver,
    write_tensor_by_layer_index: dict[LayerIndex, torch.Tensor] | dict[int, torch.Tensor],
    output_dst: DerivedScalarType,
) -> ScalarDeriver:
    """Converts a scalar deriver for a scalar type that is 1:1 with an ActivationLocationType
    to a scalar deriver for the write vector of the layer at each token."""

    check_write_tensor_device_matches(
        scalar_deriver,
        write_tensor_by_layer_index,
    )

    def multiply_by_write(
        activations: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD, "Backward pass not defined for write"
        assert (
            layer_index in write_tensor_by_layer_index
        ), f"{layer_index=} not in {write_tensor_by_layer_index.keys()=}"
        return torch.einsum(
            "ta,ao->to",
            activations,
            write_tensor_by_layer_index[layer_index],  # type: ignore
        )

    return scalar_deriver.apply_layerwise_transform_fn_to_output(
        multiply_by_write,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=output_dst,
    )


def convert_scalar_deriver_to_write_vector(
    scalar_deriver: ScalarDeriver,
    write_tensor_by_layer_index: dict[LayerIndex, torch.Tensor] | dict[int, torch.Tensor],
    output_dst: DerivedScalarType,
) -> ScalarDeriver:
    """
    Converts a scalar deriver for a scalar type that is 1:1 with an ActivationLocationType to a
    scalar deriver for the write vector of the layer at each token. Must be a scalar type that is
    related to the residual stream basis by a straightforward matmul (e.g. MLP post-activations are
    related to the residual stream basis by WeightLocationType.MLP_TO_RESIDUAL).
    """

    check_write_tensor_device_matches(
        scalar_deriver,
        write_tensor_by_layer_index,
    )

    assert scalar_deriver.dst.node_type in {
        NodeType.MLP_NEURON,
        NodeType.V_CHANNEL,
        NodeType.AUTOENCODER_LATENT,
        NodeType.MLP_AUTOENCODER_LATENT,
        NodeType.ATTENTION_AUTOENCODER_LATENT,
    }

    def multiply_by_write_vector(
        activations: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD, "Backward pass not defined for write"
        assert (
            layer_index in write_tensor_by_layer_index
        ), f"{layer_index=} not in {write_tensor_by_layer_index.keys()=}"
        return torch.einsum(
            "ta,ao->tao",
            activations,
            write_tensor_by_layer_index[layer_index],  # type: ignore
        )

    return scalar_deriver.apply_layerwise_transform_fn_to_output(
        multiply_by_write_vector,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=output_dst,
    )


def truncate_to_expected_shape(
    tensor: torch.Tensor,
    expected_shape: list[int | None],
) -> torch.Tensor:
    """
    This asserts that the tensor has the expected shape, and optionally truncates it to that shape.
    None in expected_shape means that dimension is not checked.
    """
    if expected_shape is None:
        return tensor
    for dim, real_size, expected_size in zip(
        range(len(expected_shape)), tensor.shape, expected_shape
    ):
        if expected_size is not None:
            assert (
                real_size >= expected_size
            ), f"Dimension {dim} of tensor has size {real_size} but expected size {expected_size}"
            tensor = tensor.narrow(dim, 0, expected_size)
    return tensor


def truncate_to_expected_shape_tensor_calculate_derived_scalar_fn(
    raw_activation_data_tuple: tuple[torch.Tensor, ...],
    layer_index: LayerIndex,
    pass_type: PassType,
    expected_shape: list[int | None],
) -> torch.Tensor:
    """This either:
    converts a length 1 tuple of tensors into
    a single tensor; pass_type is asserted to be PassType.FORWARD
    or
    converts a length 2 tuple of tensors, one for the forward pass and one for the backward pass,
    into the appropriate one of those two objects, depending on the pass_type argument."""
    if len(raw_activation_data_tuple) == 1:
        # in this case, only the activations at the relevant ActivationLocationType have been loaded from disk
        assert pass_type == PassType.FORWARD
        raw_activation_data = raw_activation_data_tuple[0]
        raw_activation_data = truncate_to_expected_shape(
            raw_activation_data,
            expected_shape,
        )
        return raw_activation_data
    elif len(raw_activation_data_tuple) == 2:
        # in this case, both the activations and gradients at the relevant ActivationLocationType have been loaded from disk
        raw_activation_data, raw_gradient_data = raw_activation_data_tuple
        if pass_type == PassType.FORWARD:
            raw_activation_data = truncate_to_expected_shape(
                raw_activation_data,
                expected_shape,
            )
            return raw_activation_data
        elif pass_type == PassType.BACKWARD:
            raw_gradient_data = truncate_to_expected_shape(
                raw_gradient_data,
                expected_shape,
            )
            return raw_gradient_data
        else:
            raise ValueError(f"Unknown {pass_type=}")
    else:
        raise ValueError(f"Unknown {raw_activation_data_tuple=}")


# TODO: this entire function should be simplified or deleted?
# Can possibly just use make_scalar_deriver_factory_for_activation_location_type(ActivationLocationType.LOGITS)
# in the one place it is called
def make_truncate_to_expected_shape_scalar_deriver_factory_for_dst(
    dst: DerivedScalarType,
) -> Callable[[DstConfig], ScalarDeriver]:
    """
    This is for DerivedScalarType's 1:1 with a ActivationLocationType, which can be generated from
    just the ActivationLocationType and no additional information.
    """

    untruncated_activation_location_type_by_truncated_dst = {
        DerivedScalarType.LOGITS: ActivationLocationType.LOGITS,
    }

    assert (
        dst in untruncated_activation_location_type_by_truncated_dst
    ), f"No untruncated ActivationLocationType for this DerivedScalarType: {dst}"

    activation_location_type = untruncated_activation_location_type_by_truncated_dst[dst]

    def make_scalar_deriver_fn(
        dst_config: DstConfig,
    ) -> ScalarDeriver:
        sub_scalar_sources = get_scalar_sources_for_activation_location_types(
            activation_location_type, dst_config.derive_gradients
        )

        model_context = dst_config.get_model_context()
        expected_dimensions = dst.shape_spec_per_token_sequence
        expected_shape: list[int | None] = []
        for dimension in expected_dimensions:
            if dimension.is_model_intrinsic:
                expected_shape.append(model_context.get_dim_size(dimension))
            else:
                expected_shape.append(None)

        return ScalarDeriver(
            dst=dst,
            dst_config=dst_config,
            sub_scalar_sources=sub_scalar_sources,
            tensor_calculate_derived_scalar_fn=partial(
                truncate_to_expected_shape_tensor_calculate_derived_scalar_fn,
                expected_shape=expected_shape,
            ),
        )

    return make_scalar_deriver_fn
