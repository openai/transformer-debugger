"""
This file contains functions for generating a ScalarDeriver based on a DerivedScalarType and a
DerivedScalarTypeConfig (`make_scalar_deriver`) or for convenience based on just a HookLocationType
(`make_scalar_deriver_for_hook_location_type`). It calls make_scalar_deriver_... functions defined
in other files within derived_scalars/.
"""

from typing import Callable

from neuron_explainer.activations.derived_scalars.attention import (
    make_attn_act_times_grad_per_sequence_token_scalar_deriver,
    make_attn_weighted_value_scalar_deriver,
    make_attn_write_norm_per_sequence_token_scalar_deriver,
    make_attn_write_norm_scalar_deriver,
    make_attn_write_scalar_deriver,
    make_attn_write_sum_heads_scalar_deriver,
    make_attn_write_to_final_residual_grad_per_sequence_token_scalar_deriver,
    make_attn_write_to_latent_per_sequence_token_batched_scalar_deriver,
    make_attn_write_to_latent_per_sequence_token_scalar_deriver,
    make_attn_write_to_latent_scalar_deriver,
    make_attn_write_to_latent_summed_over_heads_scalar_deriver,
    make_flattened_attn_post_softmax_act_times_grad_scalar_deriver,
    make_flattened_attn_post_softmax_scalar_deriver,
    make_flattened_attn_write_to_final_residual_grad_scalar_deriver,
    make_flattened_attn_write_to_latent_summed_over_heads_batched_scalar_deriver,
    make_flattened_attn_write_to_latent_summed_over_heads_scalar_deriver,
    make_unflattened_attn_write_norm_scalar_deriver,
    make_unflattened_attn_write_to_final_activation_residual_grad_scalar_deriver,
    make_unflattened_attn_write_to_final_residual_grad_scalar_deriver,
)
from neuron_explainer.activations.derived_scalars.autoencoder import (
    make_autoencoder_latent_grad_wrt_mlp_post_act_input_scalar_deriver,
    make_autoencoder_latent_grad_wrt_residual_input_scalar_deriver,
    make_autoencoder_latent_scalar_deriver_factory,
    make_autoencoder_write_norm_scalar_deriver_factory,
    make_online_autoencoder_act_times_grad_scalar_deriver_factory,
    make_online_autoencoder_error_scalar_deriver_factory,
    make_online_autoencoder_latent_scalar_deriver_factory,
    make_online_autoencoder_latentwise_write_scalar_deriver_factory,
    make_online_autoencoder_write_norm_scalar_deriver_factory,
    make_online_autoencoder_write_to_final_activation_residual_grad_scalar_deriver_factory,
    make_online_autoencoder_write_to_final_residual_grad_scalar_deriver_factory,
    make_online_mlp_autoencoder_error_act_times_grad_scalar_deriver,
    make_online_mlp_autoencoder_error_write_norm_scalar_deriver,
    make_online_mlp_autoencoder_error_write_to_final_residual_grad_scalar_deriver,
)
from neuron_explainer.activations.derived_scalars.config import DstConfig
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.edge_activation import (
    make_in_edge_activation_scalar_deriver_factory,
)
from neuron_explainer.activations.derived_scalars.edge_attribution import (
    make_attn_out_edge_attribution_scalar_deriver,
    make_grad_of_downstream_subnode_attribution_scalar_deriver,
    make_in_edge_attribution_scalar_deriver_factory,
    make_mlp_out_edge_attribution_scalar_deriver,
    make_node_write_scalar_deriver,
    make_node_write_to_final_residual_grad_scalar_deriver,
    make_online_autoencoder_out_edge_attribution_scalar_deriver,
    make_token_out_edge_attribution_scalar_deriver,
)
from neuron_explainer.activations.derived_scalars.mlp import (
    make_mlp_neuronwise_write_scalar_deriver,
    make_mlp_write_norm_scalar_deriver,
    make_mlp_write_to_final_activation_residual_grad_scalar_deriver,
    make_mlp_write_to_final_residual_grad_scalar_deriver,
    make_resid_delta_mlp_from_mlp_post_act_scalar_deriver,
)
from neuron_explainer.activations.derived_scalars.raw_activations import (
    make_scalar_deriver_factory_for_act_times_grad,
    make_scalar_deriver_factory_for_activation_location_type,
    make_truncate_to_expected_shape_scalar_deriver_factory_for_dst,
)
from neuron_explainer.activations.derived_scalars.residual import (
    make_previous_layer_resid_post_mlp_scalar_deriver,
    make_residual_norm_scalar_deriver_factory_for_activation_location_type,
    make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type,
    make_token_attribution_scalar_deriver,
    make_unity_scalar_deriver,
    make_vocab_token_write_to_input_direction_scalar_deriver,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import ScalarDeriver
from neuron_explainer.models.model_component_registry import ActivationLocationType, NodeType

### REGISTRY; ADD NEW TYPES HERE, AND ALSO IN ENUM IN scalar_deriver.py ###

# This contains a function to generate each implemented derived scalar type. Called by
# make_scalar_deriver below.
_DERIVED_SCALAR_TYPE_REGISTRY: dict[DerivedScalarType, Callable[[DstConfig], ScalarDeriver]] = {
    DerivedScalarType.RESID_POST_EMBEDDING: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_EMBEDDING
    ),
    DerivedScalarType.RESID_POST_MLP: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_MLP
    ),
    DerivedScalarType.RESID_POST_ATTN: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_ATTN
    ),
    DerivedScalarType.RESID_FINAL_LAYER_NORM_SCALE: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_FINAL_LAYER_NORM_SCALE
    ),
    DerivedScalarType.ATTN_INPUT_LAYER_NORM_SCALE: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_INPUT_LAYER_NORM_SCALE
    ),
    DerivedScalarType.MLP_INPUT_LAYER_NORM_SCALE: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.MLP_INPUT_LAYER_NORM_SCALE
    ),
    DerivedScalarType.LOGITS: make_truncate_to_expected_shape_scalar_deriver_factory_for_dst(
        DerivedScalarType.LOGITS
    ),
    DerivedScalarType.MLP_PRE_ACT: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.MLP_PRE_ACT
    ),
    DerivedScalarType.MLP_POST_ACT: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.MLP_POST_ACT
    ),
    DerivedScalarType.ATTN_QUERY: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_QUERY
    ),
    DerivedScalarType.ATTN_KEY: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_KEY
    ),
    DerivedScalarType.ATTN_VALUE: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_VALUE
    ),
    DerivedScalarType.ATTN_QK_LOGITS: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_QK_LOGITS
    ),
    DerivedScalarType.ATTN_QK_PROBS: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_QK_PROBS
    ),
    DerivedScalarType.ATTN_WEIGHTED_SUM_OF_VALUES: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES
    ),
    DerivedScalarType.RESID_DELTA_ATTN: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_DELTA_ATTN
    ),
    DerivedScalarType.ATTN_WRITE_NORM: make_attn_write_norm_scalar_deriver,
    DerivedScalarType.FLATTENED_ATTN_POST_SOFTMAX: make_flattened_attn_post_softmax_scalar_deriver,
    DerivedScalarType.ATTN_ACT_TIMES_GRAD: make_flattened_attn_post_softmax_act_times_grad_scalar_deriver,
    DerivedScalarType.RESID_DELTA_MLP: make_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_DELTA_MLP,
    ),
    DerivedScalarType.RESID_DELTA_MLP_FROM_MLP_POST_ACT: make_resid_delta_mlp_from_mlp_post_act_scalar_deriver,
    DerivedScalarType.MLP_WRITE_NORM: make_mlp_write_norm_scalar_deriver,
    DerivedScalarType.MLP_ACT_TIMES_GRAD: make_scalar_deriver_factory_for_act_times_grad(
        ActivationLocationType.MLP_POST_ACT,
        DerivedScalarType.MLP_ACT_TIMES_GRAD,
    ),
    DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD: make_mlp_write_to_final_residual_grad_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_NORM_PER_SEQUENCE_TOKEN: make_attn_write_norm_per_sequence_token_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD_PER_SEQUENCE_TOKEN: make_attn_write_to_final_residual_grad_per_sequence_token_scalar_deriver,
    DerivedScalarType.ATTN_ACT_TIMES_GRAD_PER_SEQUENCE_TOKEN: make_attn_act_times_grad_per_sequence_token_scalar_deriver,
    DerivedScalarType.RESID_POST_EMBEDDING_NORM: make_residual_norm_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_EMBEDDING
    ),
    DerivedScalarType.RESID_POST_MLP_NORM: make_residual_norm_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_MLP
    ),
    DerivedScalarType.RESID_POST_ATTN_NORM: make_residual_norm_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_ATTN
    ),
    DerivedScalarType.MLP_LAYER_WRITE_NORM: make_residual_norm_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_DELTA_MLP
    ),
    DerivedScalarType.ATTN_LAYER_WRITE_NORM: make_residual_norm_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_DELTA_ATTN
    ),
    DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD: make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_EMBEDDING,
        use_existing_backward_pass_for_final_residual_grad=True,
    ),
    DerivedScalarType.RESID_POST_MLP_PROJ_TO_FINAL_RESIDUAL_GRAD: make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_MLP,
        use_existing_backward_pass_for_final_residual_grad=True,
    ),
    DerivedScalarType.RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD: make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_ATTN,
        use_existing_backward_pass_for_final_residual_grad=True,
    ),
    DerivedScalarType.MLP_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD: make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_DELTA_MLP,
        use_existing_backward_pass_for_final_residual_grad=True,
    ),
    DerivedScalarType.ATTN_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD: make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_DELTA_ATTN,
        use_existing_backward_pass_for_final_residual_grad=True,
    ),
    DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD: make_scalar_deriver_factory_for_act_times_grad(
        ActivationLocationType.ATTN_QK_PROBS,
        DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD,
    ),
    DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM: make_unflattened_attn_write_norm_scalar_deriver,
    DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD: make_unflattened_attn_write_to_final_residual_grad_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD: make_flattened_attn_write_to_final_residual_grad_scalar_deriver,
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR: make_online_autoencoder_error_scalar_deriver_factory(
        ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR
    ),
    DerivedScalarType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR: make_online_autoencoder_error_scalar_deriver_factory(
        ActivationLocationType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR
    ),
    DerivedScalarType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR: make_online_autoencoder_error_scalar_deriver_factory(
        ActivationLocationType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_ACT_TIMES_GRAD: make_online_mlp_autoencoder_error_act_times_grad_scalar_deriver,
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_NORM: make_online_mlp_autoencoder_error_write_norm_scalar_deriver,
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_TO_FINAL_RESIDUAL_GRAD: make_online_mlp_autoencoder_error_write_to_final_residual_grad_scalar_deriver,
    DerivedScalarType.ATTN_WRITE: make_attn_write_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_SUM_HEADS: make_attn_write_sum_heads_scalar_deriver,
    DerivedScalarType.MLP_WRITE: make_mlp_neuronwise_write_scalar_deriver,
    DerivedScalarType.ATTN_WEIGHTED_VALUE: make_attn_weighted_value_scalar_deriver,
    DerivedScalarType.PREVIOUS_LAYER_RESID_POST_MLP: make_previous_layer_resid_post_mlp_scalar_deriver,
    DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: make_mlp_write_to_final_activation_residual_grad_scalar_deriver,
    DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: make_unflattened_attn_write_to_final_activation_residual_grad_scalar_deriver,
    DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: make_residual_projection_to_final_residual_grad_scalar_deriver_factory_for_activation_location_type(
        ActivationLocationType.RESID_POST_EMBEDDING,
        use_existing_backward_pass_for_final_residual_grad=False,
    ),
    DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT: make_autoencoder_latent_grad_wrt_residual_input_scalar_deriver,
    DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_MLP_POST_ACT_INPUT: make_autoencoder_latent_grad_wrt_mlp_post_act_input_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_TO_LATENT: make_attn_write_to_latent_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS: make_attn_write_to_latent_summed_over_heads_scalar_deriver,
    DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS: make_flattened_attn_write_to_latent_summed_over_heads_scalar_deriver,
    DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED: make_flattened_attn_write_to_latent_summed_over_heads_batched_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN: make_attn_write_to_latent_per_sequence_token_scalar_deriver,
    DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED: make_attn_write_to_latent_per_sequence_token_batched_scalar_deriver,
    DerivedScalarType.TOKEN_ATTRIBUTION: make_token_attribution_scalar_deriver,
    DerivedScalarType.SINGLE_NODE_WRITE: make_node_write_scalar_deriver,
    DerivedScalarType.GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION: make_grad_of_downstream_subnode_attribution_scalar_deriver,
    DerivedScalarType.ATTN_OUT_EDGE_ATTRIBUTION: make_attn_out_edge_attribution_scalar_deriver,
    DerivedScalarType.MLP_OUT_EDGE_ATTRIBUTION: make_mlp_out_edge_attribution_scalar_deriver,
    DerivedScalarType.ONLINE_AUTOENCODER_OUT_EDGE_ATTRIBUTION: make_online_autoencoder_out_edge_attribution_scalar_deriver,
    DerivedScalarType.ATTN_QUERY_IN_EDGE_ATTRIBUTION: make_in_edge_attribution_scalar_deriver_factory(
        NodeType.ATTENTION_HEAD, ActivationLocationType.ATTN_QUERY
    ),
    DerivedScalarType.ATTN_KEY_IN_EDGE_ATTRIBUTION: make_in_edge_attribution_scalar_deriver_factory(
        NodeType.ATTENTION_HEAD, ActivationLocationType.ATTN_KEY
    ),
    DerivedScalarType.ATTN_VALUE_IN_EDGE_ATTRIBUTION: make_in_edge_attribution_scalar_deriver_factory(
        NodeType.ATTENTION_HEAD, ActivationLocationType.ATTN_VALUE
    ),
    DerivedScalarType.MLP_IN_EDGE_ATTRIBUTION: make_in_edge_attribution_scalar_deriver_factory(
        NodeType.MLP_NEURON
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ATTRIBUTION: make_in_edge_attribution_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD: make_node_write_to_final_residual_grad_scalar_deriver,
    DerivedScalarType.TOKEN_OUT_EDGE_ATTRIBUTION: make_token_out_edge_attribution_scalar_deriver,
    DerivedScalarType.VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION: make_vocab_token_write_to_input_direction_scalar_deriver,
    DerivedScalarType.ALWAYS_ONE: make_unity_scalar_deriver,
    DerivedScalarType.ATTN_QUERY_IN_EDGE_ACTIVATION: make_in_edge_activation_scalar_deriver_factory(
        ActivationLocationType.ATTN_QK_PROBS, ActivationLocationType.ATTN_QUERY
    ),
    DerivedScalarType.ATTN_KEY_IN_EDGE_ACTIVATION: make_in_edge_activation_scalar_deriver_factory(
        ActivationLocationType.ATTN_QK_PROBS, ActivationLocationType.ATTN_KEY
    ),
    DerivedScalarType.MLP_IN_EDGE_ACTIVATION: make_in_edge_activation_scalar_deriver_factory(
        ActivationLocationType.MLP_POST_ACT,
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ACTIVATION: make_in_edge_activation_scalar_deriver_factory(
        ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
    ),
    DerivedScalarType.AUTOENCODER_LATENT: make_autoencoder_latent_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.MLP_AUTOENCODER_LATENT: make_autoencoder_latent_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ATTENTION_AUTOENCODER_LATENT: make_autoencoder_latent_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_LATENT: make_online_autoencoder_latent_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT: make_online_autoencoder_latent_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT: make_online_autoencoder_latent_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD: make_online_autoencoder_act_times_grad_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD: make_online_autoencoder_act_times_grad_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD: make_online_autoencoder_act_times_grad_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD: make_online_autoencoder_write_to_final_residual_grad_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD: make_online_autoencoder_write_to_final_residual_grad_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD: make_online_autoencoder_write_to_final_residual_grad_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: make_online_autoencoder_write_to_final_activation_residual_grad_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: make_online_autoencoder_write_to_final_activation_residual_grad_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: make_online_autoencoder_write_to_final_activation_residual_grad_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE: make_online_autoencoder_latentwise_write_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE: make_online_autoencoder_latentwise_write_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE: make_online_autoencoder_latentwise_write_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM: make_online_autoencoder_write_norm_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_MLP_AUTOENCODER_WRITE_NORM: make_online_autoencoder_write_norm_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM: make_online_autoencoder_write_norm_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
    DerivedScalarType.AUTOENCODER_WRITE_NORM: make_autoencoder_write_norm_scalar_deriver_factory(
        NodeType.AUTOENCODER_LATENT
    ),
    DerivedScalarType.MLP_AUTOENCODER_WRITE_NORM: make_autoencoder_write_norm_scalar_deriver_factory(
        NodeType.MLP_AUTOENCODER_LATENT
    ),
    DerivedScalarType.ATTENTION_AUTOENCODER_WRITE_NORM: make_autoencoder_write_norm_scalar_deriver_factory(
        NodeType.ATTENTION_AUTOENCODER_LATENT
    ),
}


def make_scalar_deriver(
    dst: DerivedScalarType,
    dst_config: DstConfig,
) -> ScalarDeriver:
    """The model name and layer indices of interest might or might not need to be specified
    based on the dst. In particular, if the dst
    is also a HookLocationType, then the model name and layer indices are not needed."""

    assert dst in _DERIVED_SCALAR_TYPE_REGISTRY, f"Unknown {dst=}"
    # this is derived from one or more HookLocationTypes, via the function
    # specified in the registry
    make_scalar_deriver_fn = _DERIVED_SCALAR_TYPE_REGISTRY[dst]

    return make_scalar_deriver_fn(dst_config)


def make_scalar_deriver_for_activation_location_type(
    activation_location_type: ActivationLocationType,
    derive_gradients: bool = False,
) -> ScalarDeriver:
    return make_scalar_deriver(
        DerivedScalarType.from_activation_location_type(activation_location_type),
        dst_config=DstConfig(
            derive_gradients=derive_gradients,
        ),
    )
