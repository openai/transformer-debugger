"""This file defines ScalarDerivers for efficiently computing the direct effect of a single upstream node
on many downstream nodes."""

from typing import Callable

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.node_write import make_node_write_scalar_source
from neuron_explainer.activations.derived_scalars.reconstituter_class import ActivationReconstituter
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DstConfig,
    ScalarDeriver,
    ScalarSource,
)
from neuron_explainer.models.model_component_registry import ActivationLocationType
from neuron_explainer.models.model_context import StandardModelContext


def convert_node_write_scalar_deriver_to_in_edge_activation(
    node_write_scalar_source: ScalarSource,
    output_dst: DerivedScalarType,
    dst_config: DstConfig,
    downstream_activation_location_type: ActivationLocationType,
    downstream_q_or_k: ActivationLocationType | None,
) -> ScalarDeriver:
    """Converts a scalar deriver for a write vector from some upstream node type to a scalar deriver for
    in edge activation for downstream nodes of some type (MLP, autoencoder, or attention head). In the
    case of attention heads, this is split up by subnode (Q or K)."""

    model_context = dst_config.get_model_context()
    autoencoder_context = dst_config.get_autoencoder_context()
    assert isinstance(model_context, StandardModelContext)
    transformer = model_context.get_or_create_model()
    reconstituter = ActivationReconstituter.from_activation_location_type(
        transformer=transformer,
        autoencoder_context=autoencoder_context,
        activation_location_type=downstream_activation_location_type,
        q_or_k=downstream_q_or_k,
    )
    return reconstituter.make_jvp_scalar_deriver(
        write_scalar_source=node_write_scalar_source,
        dst_config=dst_config,
        output_dst=output_dst,
    )


def make_in_edge_activation_scalar_deriver_factory(
    activation_location_type: ActivationLocationType,
    q_or_k: ActivationLocationType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    """Returns a function that creates a scalar deriver for the edge attribution from arbitrary node
    to the specified downstream activation location type / sub activation location type (MLP post act,
    autoencoder latent, attention head Q or K).
    """

    sub_node_type_to_output_dst = {
        (ActivationLocationType.MLP_POST_ACT, None): DerivedScalarType.MLP_IN_EDGE_ACTIVATION,
        (
            ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
            None,
        ): DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ACTIVATION,
        (
            ActivationLocationType.ATTN_QK_PROBS,
            ActivationLocationType.ATTN_QUERY,
        ): DerivedScalarType.ATTN_QUERY_IN_EDGE_ACTIVATION,
        (
            ActivationLocationType.ATTN_QK_PROBS,
            ActivationLocationType.ATTN_KEY,
        ): DerivedScalarType.ATTN_KEY_IN_EDGE_ACTIVATION,
    }

    output_dst = sub_node_type_to_output_dst[(activation_location_type, q_or_k)]

    def make_in_edge_activation_scalar_deriver(dst_config: DstConfig) -> ScalarDeriver:
        node_write_scalar_source = make_node_write_scalar_source(dst_config)
        return convert_node_write_scalar_deriver_to_in_edge_activation(
            node_write_scalar_source=node_write_scalar_source,
            output_dst=output_dst,
            dst_config=dst_config,
            downstream_activation_location_type=activation_location_type,
            downstream_q_or_k=q_or_k,
        )

    return make_in_edge_activation_scalar_deriver
