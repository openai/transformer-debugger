// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * The type of component / fundamental unit to use for Attention layers.
 *
 * This determines which types of node appear in the node table to represent the Attention layers.
 * Heads are the fundamental unit of Attention layers, but autoencoder latents are more interpretable.
 */
export enum ComponentTypeForAttention {
  ATTENTION_HEAD = "attention_head",
  AUTOENCODER_LATENT = "autoencoder_latent",
}
