"""
"Direct effects" of one node on another, or on the loss, are defined by first computing the gradient
of the downstream node's activation with respect to the residual stream immediately preceding the
downstream node. We then compute the inner product of this gradient with the write vector of the
upstream node. If the upstream node is in the residual stream basis, then it is considered to be its
own "write vector" for this purpose.

This file contains code for performing the computation described above, for upstream nodes of
various types.
"""

import dataclasses
from typing import Callable

import torch

from neuron_explainer.activations.derived_scalars.config import TraceConfig
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import AttnSubNodeIndex, NodeIndex
from neuron_explainer.activations.derived_scalars.locations import (
    ConstantLayerIndexer,
    get_previous_residual_dst_for_node_type,
    precedes_final_layer,
)
from neuron_explainer.activations.derived_scalars.raw_activations import (
    check_write_tensor_device_matches,
)
from neuron_explainer.activations.derived_scalars.reconstituted import make_apply_attn_V_act
from neuron_explainer.activations.derived_scalars.reconstituter_class import (
    Reconstituter,
    make_no_backward_pass_scalar_source_for_final_residual_grad,
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
    get_autoencoder_write_tensor_by_layer_index,
    get_mlp_write_tensor_by_layer_index,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    LayerIndex,
    NodeType,
    PassType,
)
from neuron_explainer.models.model_context import ModelContext


def make_write_to_direction_tensor_fn(
    node_type: NodeType,
    write_tensor_by_layer_index: dict[LayerIndex, torch.Tensor] | dict[int, torch.Tensor] | None,
    layer_precedes_direction_layer_fn: Callable[[LayerIndex], bool],
) -> Callable[[torch.Tensor, torch.Tensor, LayerIndex, PassType], torch.Tensor]:
    """
    To convert an "activation" tensor to a "projection of write to direction" tensor, we need to convert the
    "activation" to the residual stream basis (using a write tensor) if it is not already, and then project to
    the direction of interest. This function constructs the appropriate tensor operation to perform this projection
    based on the node_type of the activation tensor (passed as an argument). The write_tensor_by_layer_index argument
    defines the conversion to the residual stream basis, and the layer_precedes_direction_layer_fn argument is assumed
    to return True iff the derived scalar at a given layer index is upstream of the direction of interest.
    """
    match node_type:
        case NodeType.RESIDUAL_STREAM_CHANNEL:
            assert write_tensor_by_layer_index is None

            def inner_product_with_residual(
                residual: torch.Tensor,
                direction: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:  # (num_sequence_tokens, 1)
                assert pass_type == PassType.FORWARD
                assert residual.ndim == 2
                assert residual.shape == direction.shape
                if layer_precedes_direction_layer_fn(layer_index):
                    return torch.einsum("td,td->t", residual, direction)[
                        :, None
                    ]  # sum over residual stream channels
                else:
                    return torch.zeros_like(residual[:, 0:1])

            return inner_product_with_residual
        case NodeType.MLP_NEURON | NodeType.AUTOENCODER_LATENT | NodeType.MLP_AUTOENCODER_LATENT | NodeType.ATTENTION_AUTOENCODER_LATENT:
            assert write_tensor_by_layer_index is not None

            def multiply_by_projection_to_direction(
                activations: torch.Tensor,
                direction: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> (
                torch.Tensor
            ):  # (num_sequence_tokens, num_activations [i.e. num_neurons, num_latents])
                assert layer_index is not None
                assert layer_index in write_tensor_by_layer_index
                assert pass_type == PassType.FORWARD
                assert activations.ndim == direction.ndim == 2
                if layer_precedes_direction_layer_fn(layer_index):
                    write_projection = torch.einsum(
                        "ao,to->ta", write_tensor_by_layer_index[layer_index], direction
                    )
                    return activations * write_projection
                else:
                    return torch.zeros_like(activations)

            return multiply_by_projection_to_direction
        case NodeType.V_CHANNEL:
            assert write_tensor_by_layer_index is not None

            def attn_write_to_residual_direction_tensor_calculate_derived_scalar_fn(
                attn_weighted_values: torch.Tensor,
                direction: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> (
                torch.Tensor
            ):  # (num_sequence_tokens, num_heads) or (num_sequence_tokens, num_attended_to_sequence_tokens, num_heads)
                assert layer_index is not None
                assert layer_index in write_tensor_by_layer_index
                assert pass_type == PassType.FORWARD
                # one or two token dimensions, one head dimension, one value channel dimension
                assert attn_weighted_values.ndim in {3, 4}
                if layer_precedes_direction_layer_fn(layer_index):
                    return compute_attn_write_to_residual_direction_from_attn_weighted_values(
                        attn_weighted_values=attn_weighted_values,
                        residual_direction=direction,
                        W_O=write_tensor_by_layer_index[layer_index],
                        pass_type=pass_type,
                    )  # TODO: consider splitting into two cases, once we have separate node_types
                else:
                    return torch.zeros_like(attn_weighted_values[..., 0])  # sum over last dimension

            return attn_write_to_residual_direction_tensor_calculate_derived_scalar_fn
        case _:
            raise NotImplementedError(
                f"make_write_to_direction_tensor_fn not implemented for {node_type=}"
            )


def compute_attn_write_to_residual_direction_from_attn_weighted_values(
    attn_weighted_values: torch.Tensor,
    residual_direction: torch.Tensor,
    W_O: torch.Tensor,  # hdo
    pass_type: PassType,
) -> torch.Tensor:
    assert (
        pass_type == PassType.FORWARD
    ), "only forward pass implemented for now for attn write norm from weighted sum of values"
    if attn_weighted_values.ndim == 3:
        num_sequence_tokens, nheads, d_head = attn_weighted_values.shape
    else:
        assert attn_weighted_values.ndim == 4
        (
            num_sequence_tokens,
            num_attended_to_sequence_tokens,
            nheads,
            d_head,
        ) = attn_weighted_values.shape
    assert residual_direction.shape[0] == num_sequence_tokens
    _, d_model = residual_direction.shape
    assert W_O.shape == (nheads, d_head, d_model)
    W_O = W_O.to(residual_direction.dtype)
    Wo_projection = torch.einsum("hdo,to->thd", W_O, residual_direction)
    if attn_weighted_values.ndim == 3:
        v_times_Wo_projection = torch.einsum(
            "thd,thd->th", attn_weighted_values, Wo_projection
        )  # optionally either one or two token dimensions
    else:
        assert attn_weighted_values.ndim == 4
        v_times_Wo_projection = torch.einsum(
            "tuhd,thd->tuh", attn_weighted_values, Wo_projection
        )  # optionally either one or two token dimensions
    assert (v_times_Wo_projection.shape[0], v_times_Wo_projection.shape[-1]) == (
        num_sequence_tokens,
        nheads,
    )
    return v_times_Wo_projection


def convert_scalar_deriver_to_write_to_direction_with_write_tensor(
    scalar_deriver: ScalarDeriver,
    write_tensor_by_layer_index: dict[LayerIndex, torch.Tensor] | dict[int, torch.Tensor] | None,
    direction_scalar_source: ScalarSource,
    output_dst: DerivedScalarType,
) -> ScalarDeriver:
    """Takes as input a scalar deriver for a scalar activation fully defining a write direction
    (e.g. MLP activation or autoencoder but not post-softmax attention) and a scalar deriver for a direction
    in the residual stream basis. Multiplies each activation by its associated write vector and projects to the direction
    of interest."""
    if write_tensor_by_layer_index is not None:
        check_write_tensor_device_matches(
            scalar_deriver,
            write_tensor_by_layer_index,
        )

    def derived_scalar_precedes_direction_layer(layer_index: LayerIndex) -> bool:
        return precedes_final_layer(
            final_residual_location_within_layer=direction_scalar_source.location_within_layer,
            final_residual_layer_index=direction_scalar_source.layer_index,
            derived_scalar_location_within_layer=scalar_deriver.location_within_layer,
            derived_scalar_layer_index=layer_index,
        )

    write_to_direction_tensor_fn = make_write_to_direction_tensor_fn(
        node_type=scalar_deriver.dst.node_type,
        write_tensor_by_layer_index=write_tensor_by_layer_index,
        layer_precedes_direction_layer_fn=derived_scalar_precedes_direction_layer,
    )

    return scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
        write_to_direction_tensor_fn,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=output_dst,
        other_scalar_source=direction_scalar_source,
    )


def make_final_residual_grad_scalar_source(
    dst_config: DstConfig,
    use_backward_pass: bool,
) -> ScalarSource:
    """
    Many DSTs depend on the residual stream gradient at the last point in the forward pass before the point
    from which the backward pass is run. There are two ways of deriving this residual stream gradient.

    Background on backward passes:
    By default, the backward pass is run starting from some scalar function of the transformer's
    output logits. In this case, the last relevant point in the forward pass is at the very last
    residual stream location in the network (pre- final layer norm).
    A backward pass can also be run from an arbitrary activation in the network. In this case, the
    last relevant point in the forward pass is at the residual stream location immediately preceding
    the layer index of the activation from which the backward pass is run (pre- layer norm for that
    layer).
    The DstConfig object specifies whether the backward pass is the default (trace_config=None)
    or from an activation (trace_config=TraceConfig(node_index=NodeIndex(),...)).
    Note that if all you care about for a particular DST is the gradient at the last point in the forward pass
    (i.e. the first point in the backward pass), then running the full backward pass is actually wasteful.
    If you need to compute gradients with respect to many different activations, it's best just to run the very
    first part of the backward pass if possible. This is what use_backward_pass=False does.

    Two ways of deriving the residual stream gradient:
     - use_backward_pass=True: assume a literal backward pass has been run, outside the DST setup, as specified
        by the DstConfig object. In this case, you can directly use the "raw" residual stream gradient at a location
        inferrable from dst_config.trace_config.
     - use_backward_pass=False: (specific to the case where trace_config is not None)
        do not make assumptions about the literal backward pass that has been run. Take
        the residual stream **activations** (the forward pass) at the location implied by
        dst_config.trace_config. Recompute the activation specified from those residual
        stream activations, and run a small backward pass on the activation, back to those residual stream
        activations.
    """
    if use_backward_pass:
        return make_backward_pass_scalar_source_for_final_residual_grad(dst_config)
    else:
        return make_no_backward_pass_scalar_source_for_final_residual_grad(dst_config)


def make_backward_pass_scalar_source_for_final_residual_grad(
    dst_config: DstConfig,
) -> ScalarSource:
    """Called by other make_scalar_deriver functions; not needed as a derived scalar on its own.
    Note that the dst_config is not used for the (temporary) ScalarDeriver that is returned
    by this function. This determines the config needed for a final_residual_grad scalar deriver, based on
    the config of the scalar deriver for the activation which will be multiplied by the final residual grad.
    """
    if (
        dst_config.trace_config is not None
        and dst_config.trace_config.node_type.is_autoencoder_latent
    ):
        autoencoder_dst = dst_config.get_autoencoder_dst(dst_config.trace_config.node_type)
    else:
        autoencoder_dst = None
    return make_backward_pass_scalar_source_for_final_residual_grad_helper(
        n_layers=dst_config.get_n_layers(),
        trace_config=dst_config.trace_config,
        autoencoder_dst=autoencoder_dst,
    )


def make_backward_pass_scalar_source_for_fake_final_residual_grad(
    dst_config: DstConfig,
) -> ScalarSource:
    """Called by other make_scalar_deriver functions; not needed as a derived scalar on its own.
    Note that the dst_config is not used for the (temporary) ScalarDeriver that is returned
    by this function. This determines the config needed for a final_fake_residual_grad scalar deriver, based on
    the config of the scalar deriver for the activation which will be multiplied by the final fake residual grad.

    The gradient is "fake" in the sense that a real backward pass is run from a later point in the network, but the
    gradient is assumed to be ablated such that a real gradient of interest can be computed at the residual stream
    immediately preceding the layer_index of dst_config.activation_index_for_fake_grad.
    """
    assert dst_config.activation_index_for_fake_grad is not None
    if (
        dst_config.trace_config is not None
        and dst_config.trace_config.node_type.is_autoencoder_latent
    ):
        autoencoder_dst = dst_config.get_autoencoder_dst(dst_config.trace_config.node_type)
    else:
        autoencoder_dst = None
    return make_backward_pass_scalar_source_for_final_residual_grad_helper(
        n_layers=dst_config.get_n_layers(),
        trace_config=TraceConfig.from_activation_index(
            activation_index=dst_config.activation_index_for_fake_grad
        ),
        autoencoder_dst=autoencoder_dst,
    )


def make_backward_pass_scalar_source_for_final_residual_grad_helper(
    n_layers: int,  # total layers in model
    trace_config: TraceConfig | None,
    autoencoder_dst: DerivedScalarType | None,
) -> ScalarSource:
    """
    Returns the location of the last residual stream location prior to the layer norm preceding the location from
    which .backward() is being computed
    """
    # lazily avoid circular import
    from neuron_explainer.activations.derived_scalars.make_scalar_derivers import (
        make_scalar_deriver,
    )

    if trace_config is None:
        return RawScalarSource(
            activation_location_type=ActivationLocationType.RESID_POST_MLP,
            pass_type=PassType.BACKWARD,
            layer_indexer=ConstantLayerIndexer(n_layers - 1),
        )
    else:
        layer_index = trace_config.layer_index
        assert layer_index is not None
        residual_dst = get_previous_residual_dst_for_node_type(
            node_type=trace_config.node_type,
            autoencoder_dst=autoencoder_dst,
        )
        return DerivedScalarSource(
            scalar_deriver=make_scalar_deriver(
                residual_dst, DstConfig(layer_indices=[layer_index], derive_gradients=True)
            ),
            pass_type=PassType.BACKWARD,
            layer_indexer=ConstantLayerIndexer(layer_index),
        )


def convert_scalar_deriver_to_write_to_final_residual_grad(
    scalar_deriver: ScalarDeriver,
    output_dst: DerivedScalarType,
    use_existing_backward_pass_for_final_residual_grad: bool,
) -> ScalarDeriver:
    direction_scalar_source = make_final_residual_grad_scalar_source(
        scalar_deriver.dst_config, use_existing_backward_pass_for_final_residual_grad
    )
    return convert_scalar_deriver_to_write_to_direction(
        scalar_deriver=scalar_deriver,
        direction_scalar_source=direction_scalar_source,
        output_dst=output_dst,
    )


def convert_scalar_deriver_to_write_to_direction(
    scalar_deriver: ScalarDeriver,
    direction_scalar_source: ScalarSource,
    output_dst: DerivedScalarType,
) -> ScalarDeriver:
    model_context = scalar_deriver.dst_config.get_model_context()
    layer_indices = scalar_deriver.dst_config.layer_indices or list(range(model_context.n_layers))
    node_type = scalar_deriver.dst.node_type
    match node_type:
        case NodeType.RESIDUAL_STREAM_CHANNEL:
            write_tensor_by_layer_index: dict[LayerIndex, torch.Tensor] | None = None
        case NodeType.MLP_NEURON:
            write_tensor_by_layer_index = get_mlp_write_tensor_by_layer_index(
                model_context=model_context,
                layer_indices=layer_indices,
            )
        case NodeType.V_CHANNEL:
            write_tensor_by_layer_index = get_attn_write_tensor_by_layer_index(
                model_context=model_context,
                layer_indices=layer_indices,
            )
        case (
            NodeType.AUTOENCODER_LATENT
            | NodeType.MLP_AUTOENCODER_LATENT
            | NodeType.ATTENTION_AUTOENCODER_LATENT
        ):
            autoencoder_context = scalar_deriver.dst_config.get_autoencoder_context(node_type)
            assert autoencoder_context is not None
            write_tensor_by_layer_index = get_autoencoder_write_tensor_by_layer_index(
                model_context=model_context,
                autoencoder_context=autoencoder_context,
            )
        case _:
            raise NotImplementedError(
                f"convert_scalar_deriver_to_write_to_direction not implemented for {node_type=}"
            )
    return convert_scalar_deriver_to_write_to_direction_with_write_tensor(
        scalar_deriver=scalar_deriver,
        write_tensor_by_layer_index=write_tensor_by_layer_index,
        direction_scalar_source=direction_scalar_source,
        output_dst=output_dst,
    )


def make_reconstituted_attention_direct_effect_fn(
    model_context: ModelContext,
    layer_indices: list[int] | None,
    detach_layer_norm_scale: bool,
) -> Callable[[torch.Tensor, torch.Tensor, LayerIndex, PassType], torch.Tensor]:
    apply_attn_V_act = make_apply_attn_V_act(
        transformer=model_context.get_or_create_model(),
        q_k_or_v=ActivationLocationType.ATTN_VALUE,
        detach_layer_norm_scale=detach_layer_norm_scale,
    )

    write_tensor_by_layer_index = get_attn_write_tensor_by_layer_index(
        model_context=model_context,
        layer_indices=layer_indices,
    )

    def direct_effect_fn(
        resid: torch.Tensor,
        grad: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        # grad is a d_model-dimensional vector
        attn, V = apply_attn_V_act(resid, layer_index, pass_type)
        attn_weighted_V = torch.einsum("qkh,khd->qkhd", attn, V)
        grad_proj_to_V = torch.einsum("hdo,qo->qhd", write_tensor_by_layer_index[layer_index], grad)
        # grad is w/r/t (attn_weighted_V summed over k, or ATTN_WEIGHTED_SUM_OF_VALUES)
        return torch.einsum("qkhd,qhd->qkh", attn_weighted_V, grad_proj_to_V)

    return direct_effect_fn


class AttentionDirectEffectReconstituter(Reconstituter):
    """Reconstitute an attention head's write to a particular direction"""

    requires_other_scalar_source = True
    node_type = NodeType.ATTENTION_HEAD

    def __init__(
        self,
        model_context: ModelContext,
        layer_indices: list[int] | None,
        detach_layer_norm_scale: bool,
    ):
        super().__init__()
        self._reconstitute_activations_fn = make_reconstituted_attention_direct_effect_fn(
            model_context=model_context,
            layer_indices=layer_indices,
            detach_layer_norm_scale=detach_layer_norm_scale,
        )
        self._layer_indices = layer_indices
        self.residual_dst = get_previous_residual_dst_for_node_type(
            node_type=self.node_type,
            autoencoder_dst=None,
        )

    def reconstitute_activations(
        self,
        resid: torch.Tensor,
        grad: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD
        assert grad is not None
        return self._reconstitute_activations_fn(
            resid,
            grad,
            layer_index,
            pass_type,
        )

    def make_other_scalar_source(self, dst_config: DstConfig) -> ScalarSource:
        # make_backward_pass_scalar_source_for_final_residual_grad
        # does not use most of the fields of dst_config; just
        # get_n_layers(), get_autoencoder_dst(), and trace_config
        return make_backward_pass_scalar_source_for_final_residual_grad(dst_config)

    def _check_node_index(self, node_index: NodeIndex) -> None:
        assert node_index.node_type == self.node_type
        assert node_index.pass_type == PassType.FORWARD
        assert node_index.layer_index is not None
        # self._layer_indices = None -> support all layer_indices; otherwise only a subset
        # of layer indices are loaded
        assert self._layer_indices is None or node_index.layer_index in self._layer_indices
        if isinstance(node_index, AttnSubNodeIndex):
            assert node_index.q_k_or_v == ActivationLocationType.ATTN_VALUE

    def make_scalar_hook_for_node_index(
        self, node_index: NodeIndex
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        self._check_node_index(node_index)
        assert node_index.ndim == 0

        def get_activation_from_layer_activations(layer_activations: torch.Tensor) -> torch.Tensor:
            return layer_activations[node_index.tensor_indices]

        return get_activation_from_layer_activations

    def make_gradient_scalar_deriver_for_node_index(
        self,
        node_index: NodeIndex,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> ScalarDeriver:
        self._check_node_index(node_index)
        assert node_index.layer_index is not None
        dst_config_for_layer = dataclasses.replace(
            dst_config,
            layer_indices=[node_index.layer_index],
        )
        scalar_hook = self.make_scalar_hook_for_node_index(node_index)
        return self.make_gradient_scalar_deriver(
            scalar_hook=scalar_hook,
            dst_config=dst_config_for_layer,
            output_dst=output_dst,
        )

    def make_gradient_scalar_source_for_node_index(
        self,
        node_index: NodeIndex,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> DerivedScalarSource:
        scalar_hook = self.make_scalar_hook_for_node_index(node_index)
        gradient_scalar_deriver = self.make_gradient_scalar_deriver(
            scalar_hook=scalar_hook,
            dst_config=dst_config,
            output_dst=output_dst,
        )
        assert node_index.layer_index is not None
        return DerivedScalarSource(
            scalar_deriver=gradient_scalar_deriver,
            pass_type=PassType.FORWARD,
            layer_indexer=ConstantLayerIndexer(node_index.layer_index),
        )
