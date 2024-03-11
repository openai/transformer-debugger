// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Dimensions correspond to the names of dimensions of activation tensors, and can depend on the input,
 * the model, or e.g. parameters of added subgraphs such as autoencoders.
 * The dimensions below are taken to be 'per layer' wherever applicable.
 * Dimensions associated with attention heads (e.g. value channels) are taken to be 'per attention head'.
 */
export enum Dimension {
  SEQUENCE_TOKENS = "sequence_tokens",
  ATTENDED_TO_SEQUENCE_TOKENS = "attended_to_sequence_tokens",
  MAX_CONTEXT_LENGTH = "max_context_length",
  RESIDUAL_STREAM_CHANNELS = "residual_stream_channels",
  VOCAB_SIZE = "vocab_size",
  ATTN_HEADS = "attn_heads",
  QUERY_AND_KEY_CHANNELS = "query_and_key_channels",
  VALUE_CHANNELS = "value_channels",
  MLP_ACTS = "mlp_acts",
  LAYERS = "layers",
  SINGLETON = "singleton",
  AUTOENCODER_LATENTS = "autoencoder_latents",
  AUTOENCODER_LATENTS_BY_TOKEN_PAIR = "autoencoder_latents_by_token_pair",
}
