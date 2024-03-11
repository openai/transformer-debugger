"""This file defines ScalarDerivers for a single node's residual stream write vector."""

import dataclasses

import torch

from neuron_explainer.activations.derived_scalars.attention import (
    make_attn_weighted_value_scalar_deriver,
)
from neuron_explainer.activations.derived_scalars.autoencoder import (
    make_online_autoencoder_latent_scalar_deriver_factory,
)
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    make_python_slice_from_tensor_indices,
)
from neuron_explainer.activations.derived_scalars.locations import ConstantLayerIndexer
from neuron_explainer.activations.derived_scalars.mlp import get_base_mlp_scalar_deriver
from neuron_explainer.activations.derived_scalars.postprocessing import ResidualWriteConverter
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DerivedScalarSource,
    DstConfig,
    ScalarDeriver,
)
from neuron_explainer.models.autoencoder_context import MultiAutoencoderContext
from neuron_explainer.models.model_component_registry import NodeType, PassType


def make_node_write_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the write vector from some upstream node type (MLP, autoencoder, or attention head)."""
    node_index = dst_config.node_index_for_attribution
    assert node_index is not None
    assert node_index.layer_index is not None
    model_context = dst_config.get_model_context()
    autoencoder_context = dst_config.get_autoencoder_context()
    multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
        autoencoder_context
    )
    residual_write_converter = ResidualWriteConverter(
        model_context=model_context,
        multi_autoencoder_context=multi_autoencoder_context,
    )  # though called a Postprocessor, this converter is being used as part of the computation of this DST
    # It knows how to generate a residual stream write vector for a single node, and skips out on generating
    # residual stream write vectors for the entire layer worth of nodes, which is a much bigger/unnecessary matmul.
    dst_config_for_attribution = dataclasses.replace(
        dst_config,
        layer_indices=[node_index.layer_index],
    )
    match node_index.node_type:
        case NodeType.ATTENTION_HEAD:
            activation_scalar_deriver = make_attn_weighted_value_scalar_deriver(
                dst_config=dst_config_for_attribution,
            )
        case NodeType.MLP_NEURON:
            activation_scalar_deriver = get_base_mlp_scalar_deriver(
                dst_config=dst_config_for_attribution,
            )
        case (
            NodeType.AUTOENCODER_LATENT
            | NodeType.MLP_AUTOENCODER_LATENT
            | NodeType.ATTENTION_AUTOENCODER_LATENT
        ):
            activation_scalar_deriver = make_online_autoencoder_latent_scalar_deriver_factory(
                node_index.node_type
            )(dst_config_for_attribution)
    ds_index = residual_write_converter.convert_node_index_to_ds_index(node_index)
    sequence_token_index = ds_index.tensor_indices[0]
    slices_for_ds_index = make_python_slice_from_tensor_indices(ds_index.tensor_indices)

    def convert_activations_to_node_write(
        activations: torch.Tensor,
        layer_index: int | None,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD
        assert layer_index == node_index.layer_index
        single_node_write = residual_write_converter.postprocess_tensor(
            ds_index,
            activations[slices_for_ds_index],
        )
        num_sequence_tokens = activations.shape[0]
        single_node_write_with_zeros = torch.zeros(
            (num_sequence_tokens,) + single_node_write.shape, device=single_node_write.device
        )
        single_node_write_with_zeros[sequence_token_index, :] = single_node_write
        return single_node_write_with_zeros

    return activation_scalar_deriver.apply_layerwise_transform_fn_to_output(
        convert_activations_to_node_write,
        pass_type_to_transform=PassType.FORWARD,
        output_dst=DerivedScalarType.SINGLE_NODE_WRITE,
    )


def make_node_write_scalar_source(
    dst_config: DstConfig,
) -> DerivedScalarSource:
    assert dst_config.node_index_for_attribution is not None
    layer_index = dst_config.node_index_for_attribution.layer_index
    assert layer_index is not None
    node_write_scalar_deriver = make_node_write_scalar_deriver(dst_config)
    return DerivedScalarSource(
        scalar_deriver=node_write_scalar_deriver,
        pass_type=PassType.FORWARD,
        layer_indexer=ConstantLayerIndexer(layer_index),
    )
