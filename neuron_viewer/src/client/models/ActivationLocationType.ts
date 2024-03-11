// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * These are the names of activations expected to be instantiated during a forward pass. All activations are
 * pre-layer norm unless otherwise specified (RESID_POST_XYZ_LAYER_NORM).
 */
export enum ActivationLocationType {
  RESID_POST_EMB = "resid.post_emb",
  RESID_DELTA_ATTN = "resid.delta_attn",
  RESID_POST_ATTN = "resid.post_attn",
  RESID_DELTA_MLP = "resid.delta_mlp",
  RESID_POST_MLP = "resid.post_mlp",
  RESID_POST_MLP_LN = "resid.post_mlp_ln",
  RESID_POST_ATTN_LN = "resid.post_attn_ln",
  RESID_POST_LN_F = "resid.post_ln_f",
  MLP_LN_SCALE = "mlp_ln.scale",
  ATTN_LN_SCALE = "attn_ln.scale",
  RESID_LN_F_SCALE = "resid.ln_f.scale",
  ATTN_Q = "attn.q",
  ATTN_K = "attn.k",
  ATTN_V = "attn.v",
  ATTN_QK_LOGITS = "attn.qk_logits",
  ATTN_QK_PROBS = "attn.qk_probs",
  ATTN_V_OUT = "attn.v_out",
  MLP_PRE_ACT = "mlp.pre_act",
  MLP_POST_ACT = "mlp.post_act",
  LOGITS = "logits",
  ONLINE_AUTOENCODER_LATENT = "online_autoencoder_latent",
  ONLINE_MLP_AUTOENCODER_LATENT = "online_mlp_autoencoder_latent",
  ONLINE_ATTENTION_AUTOENCODER_LATENT = "online_attention_autoencoder_latent",
  ONLINE_MLP_AUTOENCODER_ERROR = "online_mlp_autoencoder_error",
  ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR = "online_residual_mlp_autoencoder_error",
  ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR = "online_residual_attention_autoencoder_error",
}
