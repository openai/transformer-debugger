// TODO: Make this explanation clearer. Does this only cover the direct effect as opposed to indirect effects?
export const WRITE_MAGNITUDE_EXPLANATION =
  "Magnitude of the write vector to the direction of interest produced by the component.";

export const ACTIVATION_EXPLANATION =
  "MLP post-activation, attention post-softmax, or autoencoder latent activation.";

// TODO: Make this explanation clearer. Is this a magnitude? Does this only cover the direct effect as opposed to indirect effects?
export const DIRECTION_WRITE_EXPLANATION =
  "Direction write: Value of the write to the direction of interest.";

export const ACT_TIMES_GRAD_EXPLANATION =
  "Activation * gradient: Estimate of the total effect of the component on the activation of the direction of interest, including indirect effects through other components.";

export const TOKEN_ATTENDED_TO_EXPLANATION =
  "Token attended-to, for attention heads only, where activations are specific to a token pair. This is the least recent token in the token pair.";

export const TOKEN_ATTRIBUTED_TO_EXPLANATION =
  "Token attended-to, for attention-write autoencoder latents only. This is the token with the most positive attribution to the latent activation.";

export const TOKEN_ATTENDED_FROM_EXPLANATION =
  "Current token, for all components. For MLP neurons and MLP latents, this is the token where the component activates. For attention heads, where activations are specific to a token pair, this is the most recent token in the token pair.";
