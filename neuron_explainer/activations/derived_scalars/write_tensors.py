import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.models.autoencoder_context import (
    AutoencoderContext,
    get_autoencoder_output_weight_by_layer_index,
)
from neuron_explainer.models.model_component_registry import (
    LayerIndex,
    NodeType,
    WeightLocationType,
)
from neuron_explainer.models.model_context import ModelContext


def get_attn_write_tensor_by_layer_index(
    model_context: ModelContext,
    layer_indices: list[int] | None,
) -> dict[LayerIndex, torch.Tensor]:
    """Returns a dictionary mapping layer index to the write weight matrix for that layer."""
    if layer_indices is None:
        layer_indices = list(range(model_context.n_layers))
    W_out_by_layer_index: dict[LayerIndex, torch.Tensor] = {
        layer_index: model_context.get_weight(
            location_type=WeightLocationType.ATTN_TO_RESIDUAL,
            layer=layer_index,
            device=model_context.device,
        )  # shape (n_heads, d_head, d_model)
        for layer_index in layer_indices
    }
    return W_out_by_layer_index


def get_mlp_write_tensor_by_layer_index(
    model_context: ModelContext, layer_indices: list[int] | None
) -> dict[LayerIndex, torch.Tensor]:
    if layer_indices is None:
        layer_indices = list(range(model_context.n_layers))
    W_out_location_type = WeightLocationType.MLP_TO_RESIDUAL
    W_out_by_layer_index: dict[LayerIndex, torch.Tensor] = {
        layer_index: model_context.get_weight(
            location_type=W_out_location_type,
            layer=layer_index,
            device=model_context.device,
        )  # shape (d_ff, d_model)
        for layer_index in layer_indices
    }
    return W_out_by_layer_index


def _assert_non_none(x: LayerIndex) -> int:
    assert x is not None
    return x


def get_autoencoder_write_tensor_by_layer_index(
    autoencoder_context: AutoencoderContext,
    model_context: ModelContext,
) -> dict[LayerIndex, torch.Tensor]:
    if autoencoder_context.dst == DerivedScalarType.MLP_POST_ACT:
        autoencoder_output_weight_by_layer_index = get_autoencoder_output_weight_by_layer_index(
            autoencoder_context
        )
        W_out_by_layer_index = get_mlp_write_tensor_by_layer_index_with_autoencoder_context(
            autoencoder_context, model_context
        )
        return {
            _assert_non_none(layer_index): torch.einsum(
                "an,nd->ad",
                autoencoder_output_weight_by_layer_index[layer_index],
                W_out_by_layer_index[_assert_non_none(layer_index)],
            )
            for layer_index in autoencoder_context.layer_indices
        }
    else:
        assert (
            autoencoder_context.dst.node_type == NodeType.RESIDUAL_STREAM_CHANNEL
        ), autoencoder_context.dst
        return get_autoencoder_output_weight_by_layer_index(autoencoder_context)


def get_mlp_write_tensor_by_layer_index_with_autoencoder_context(
    autoencoder_context: AutoencoderContext,
    model_context: ModelContext,
) -> dict[int, torch.Tensor]:
    assert all(layer_index is not None for layer_index in autoencoder_context.layer_indices)
    layer_indices: list[int] = list(autoencoder_context.layer_indices)  # type: ignore
    write_tensor_by_layer_index = get_mlp_write_tensor_by_layer_index(
        model_context=model_context, layer_indices=layer_indices
    )
    return {
        _assert_non_none(layer_index): write_tensor_by_layer_index[layer_index]
        for layer_index in autoencoder_context.layer_indices
    }
