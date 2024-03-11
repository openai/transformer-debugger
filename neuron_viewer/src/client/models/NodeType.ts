// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * A "node" is defined as a model component associated with a scalar activation per
 * token or per token pair. The canonical example is an MLP neuron. An activation
 * for which the NodeType is defined has the node as the last dimension of the
 * activation tensor.
 */
export enum NodeType {
  ATTENTION_HEAD = "attention_head",
  QK_CHANNEL = "qk_channel",
  V_CHANNEL = "v_channel",
  MLP_NEURON = "mlp_neuron",
  AUTOENCODER_LATENT = "autoencoder_latent",
  MLP_AUTOENCODER_LATENT = "mlp_autoencoder_latent",
  ATTENTION_AUTOENCODER_LATENT = "attention_autoencoder_latent",
  AUTOENCODER_LATENT_BY_TOKEN_PAIR = "autoencoder_latent_by_token_pair",
  LAYER = "layer",
  RESIDUAL_STREAM_CHANNEL = "residual_stream_channel",
  VOCAB_TOKEN = "vocab_token",
}
