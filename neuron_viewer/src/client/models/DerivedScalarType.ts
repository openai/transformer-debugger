// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * List of implemented derived location types. When implementing a new one, add a make_<dst>
 * function to _DERIVED_SCALAR_TYPE_REGISTRY in make_scalar_derivers.py and add the name of the
 * derived location type to this enum.
 *
 * If implementing a new HookLocationType, also add its DerivedScalarType (trivially computed from
 * the activations) to this enum, and add a row like this to the registry:
 * DerivedScalarType.NEW_HOOK_LOCATION_TYPE: make_scalar_deriver_factory_for_hook_location_type(
 * "new_hook_location_type"
 * )
 *
 * Activations of DerivedScalarTypes for a given token sequence either have one or two dimensions
 * indexed by tokens:
 * 1. sequence_tokens: all activations have as their first dimension the number of tokens in the
 * sequence
 * 2. attended_to_sequence_tokens: pre- and post-softmax attention, and other activations derived
 * from those, have an additional token dimension, corresponding to "attended to" tokens
 * (sequence_tokens being the "attended from" tokens).
 *
 * The token dimensions can in general be represented as a (num_sequence_tokens,
 * num_sequence_tokens) matrix, but in some settings this might be represented as a non-square
 * matrix (num_sequence_tokens != num_attended_to_sequence_tokens), e.g. if there are irrelevant
 * padding tokens we wish to leave out.
 *
 * For DerivedScalarTypes not using attended_to_sequence_tokens, a
 * num_attended_to_sequence_tokens=None argument still gets passed around in computing the expected
 * shape of the activations. This argument gets ignored.
 */
export enum DerivedScalarType {
  LOGITS = "logits",
  RESID_POST_EMBEDDING = "resid_post_embedding",
  MLP_PRE_ACT = "mlp_pre_act",
  MLP_POST_ACT = "mlp_post_act",
  RESID_DELTA_MLP = "resid_delta_mlp",
  RESID_POST_MLP = "resid_post_mlp",
  ATTN_QUERY = "attn_query",
  ATTN_KEY = "attn_key",
  ATTN_VALUE = "attn_value",
  ATTN_QK_LOGITS = "attn_qk_logits",
  ATTN_QK_PROBS = "attn_qk_probs",
  ATTN_WEIGHTED_SUM_OF_VALUES = "attn_weighted_sum_of_values",
  RESID_DELTA_ATTN = "resid_delta_attn",
  RESID_POST_ATTN = "resid_post_attn",
  RESID_FINAL_LAYER_NORM_SCALE = "resid_final_layer_norm_scale",
  ATTN_INPUT_LAYER_NORM_SCALE = "attn_input_layer_norm_scale",
  MLP_INPUT_LAYER_NORM_SCALE = "mlp_input_layer_norm_scale",
  ONLINE_AUTOENCODER_LATENT = "online_autoencoder_latent",
  ATTN_WRITE_NORM = "attn_write_norm",
  FLATTENED_ATTN_POST_SOFTMAX = "flattened_attn_post_softmax",
  ATTN_ACT_TIMES_GRAD = "attn_act_times_grad",
  RESID_DELTA_MLP_FROM_MLP_POST_ACT = "resid_delta_mlp_from_mlp_post_act",
  MLP_WRITE_NORM = "mlp_write_norm",
  MLP_ACT_TIMES_GRAD = "mlp_act_times_grad",
  AUTOENCODER_LATENT = "autoencoder_latent",
  AUTOENCODER_WRITE_NORM = "autoencoder_write_norm",
  MLP_WRITE_TO_FINAL_RESIDUAL_GRAD = "mlp_write_to_final_residual_grad",
  ATTN_WRITE_NORM_PER_SEQUENCE_TOKEN = "attn_write_norm_per_sequence_token",
  ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD_PER_SEQUENCE_TOKEN = "attn_write_to_final_residual_grad_per_sequence_token",
  ATTN_ACT_TIMES_GRAD_PER_SEQUENCE_TOKEN = "attn_act_times_grad_per_sequence_token",
  RESID_POST_EMBEDDING_NORM = "resid_post_embedding_norm",
  RESID_POST_MLP_NORM = "resid_post_mlp_norm",
  MLP_LAYER_WRITE_NORM = "mlp_layer_write_norm",
  RESID_POST_ATTN_NORM = "resid_post_attn_norm",
  ATTN_LAYER_WRITE_NORM = "attn_layer_write_norm",
  RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD = "resid_post_embedding_proj_to_final_residual_grad",
  RESID_POST_MLP_PROJ_TO_FINAL_RESIDUAL_GRAD = "resid_post_mlp_proj_to_final_residual_grad",
  MLP_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD = "mlp_layer_write_to_final_residual_grad",
  RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD = "resid_post_attn_proj_to_final_residual_grad",
  ATTN_LAYER_WRITE_TO_FINAL_RESIDUAL_GRAD = "attn_layer_write_to_final_residual_grad",
  UNFLATTENED_ATTN_ACT_TIMES_GRAD = "unflattened_attn_act_times_grad",
  UNFLATTENED_ATTN_WRITE_NORM = "unflattened_attn_write_norm",
  UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD = "unflattened_attn_write_to_final_residual_grad",
  ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD = "attn_write_to_final_residual_grad",
  ONLINE_AUTOENCODER_ACT_TIMES_GRAD = "online_autoencoder_act_times_grad",
  ONLINE_AUTOENCODER_WRITE_NORM = "online_autoencoder_write_norm",
  ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD = "online_autoencoder_write_to_final_residual_grad",
  ONLINE_MLP_AUTOENCODER_ERROR = "online_mlp_autoencoder_error",
  ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR = "online_residual_mlp_autoencoder_error",
  ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR = "online_residual_attention_autoencoder_error",
  ONLINE_MLP_AUTOENCODER_ERROR_ACT_TIMES_GRAD = "online_mlp_autoencoder_error_act_times_grad",
  ONLINE_MLP_AUTOENCODER_ERROR_WRITE_NORM = "online_mlp_autoencoder_error_write_norm",
  ONLINE_MLP_AUTOENCODER_ERROR_WRITE_TO_FINAL_RESIDUAL_GRAD = "online_mlp_autoencoder_error_write_to_final_residual_grad",
  ATTN_WRITE = "attn_write",
  ATTN_WRITE_SUM_HEADS = "attn_write_sum_heads",
  MLP_WRITE = "mlp_write",
  ONLINE_AUTOENCODER_WRITE = "online_autoencoder_write",
  ATTN_WEIGHTED_VALUE = "attn_weighted_value",
  PREVIOUS_LAYER_RESID_POST_MLP = "previous_layer_resid_post_mlp",
  MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = "mlp_write_to_final_activation_residual_grad",
  UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = "unflattened_attn_write_to_final_activation_residual_grad",
  ONLINE_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = "online_autoencoder_write_to_final_activation_residual_grad",
  RESID_POST_EMBEDDING_PROJ_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = "resid_post_embedding_proj_to_final_activation_residual_grad",
  AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT = "autoencoder_latent_grad_wrt_residual_input",
  AUTOENCODER_LATENT_GRAD_WRT_MLP_POST_ACT_INPUT = "autoencoder_latent_grad_wrt_mlp_post_act_input",
  ATTN_WRITE_TO_LATENT = "attn_write_to_latent",
  ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS = "attn_write_to_latent_summed_over_heads",
  FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS = "flattened_attn_write_to_latent_summed_over_heads",
  FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED = "flattened_attn_write_to_latent_summed_over_heads_batched",
  ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN = "attn_write_to_latent_per_sequence_token",
  ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED = "attn_write_to_latent_per_sequence_token_batched",
  TOKEN_ATTRIBUTION = "token_attribution",
  SINGLE_NODE_WRITE = "single_node_write",
  GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION = "grad_of_single_subnode_attribution",
  ATTN_OUT_EDGE_ATTRIBUTION = "attn_out_edge_attribution",
  MLP_OUT_EDGE_ATTRIBUTION = "mlp_out_edge_attribution",
  ONLINE_AUTOENCODER_OUT_EDGE_ATTRIBUTION = "online_autoencoder_out_edge_attribution",
  ATTN_QUERY_IN_EDGE_ATTRIBUTION = "attn_query_in_edge_attribution",
  ATTN_KEY_IN_EDGE_ATTRIBUTION = "attn_key_in_edge_attribution",
  ATTN_VALUE_IN_EDGE_ATTRIBUTION = "attn_value_in_edge_attribution",
  MLP_IN_EDGE_ATTRIBUTION = "mlp_in_edge_attribution",
  ONLINE_AUTOENCODER_IN_EDGE_ATTRIBUTION = "online_autoencoder_in_edge_attribution",
  TOKEN_OUT_EDGE_ATTRIBUTION = "token_out_edge_attribution",
  SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD = "single_node_write_to_final_residual_grad",
  VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION = "vocab_token_write_to_input_direction",
  ALWAYS_ONE = "always_one",
  ATTN_QUERY_IN_EDGE_ACTIVATION = "attn_query_in_edge_activation",
  ATTN_KEY_IN_EDGE_ACTIVATION = "attn_key_in_edge_activation",
  MLP_IN_EDGE_ACTIVATION = "mlp_in_edge_activation",
  ONLINE_AUTOENCODER_IN_EDGE_ACTIVATION = "online_autoencoder_in_edge_activation",
  MLP_AUTOENCODER_LATENT = "mlp_autoencoder_latent",
  MLP_AUTOENCODER_WRITE_NORM = "mlp_autoencoder_write_norm",
  ONLINE_MLP_AUTOENCODER_LATENT = "online_mlp_autoencoder_latent",
  ONLINE_MLP_AUTOENCODER_WRITE = "online_mlp_autoencoder_write",
  ONLINE_MLP_AUTOENCODER_WRITE_NORM = "online_mlp_autoencoder_write_norm",
  ONLINE_MLP_AUTOENCODER_ACT_TIMES_GRAD = "online_mlp_autoencoder_act_times_grad",
  ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD = "online_mlp_autoencoder_write_to_final_residual_grad",
  ONLINE_MLP_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = "online_mlp_autoencoder_write_to_final_activation_residual_grad",
  ATTENTION_AUTOENCODER_LATENT = "attention_autoencoder_latent",
  ATTENTION_AUTOENCODER_WRITE_NORM = "attention_autoencoder_write_norm",
  ONLINE_ATTENTION_AUTOENCODER_LATENT = "online_attention_autoencoder_latent",
  ONLINE_ATTENTION_AUTOENCODER_WRITE = "online_attention_autoencoder_write",
  ONLINE_ATTENTION_AUTOENCODER_WRITE_NORM = "online_attention_autoencoder_write_norm",
  ONLINE_ATTENTION_AUTOENCODER_ACT_TIMES_GRAD = "online_attention_autoencoder_act_times_grad",
  ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD = "online_attention_autoencoder_write_to_final_residual_grad",
  ONLINE_ATTENTION_AUTOENCODER_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD = "online_attention_autoencoder_write_to_final_activation_residual_grad",
}
