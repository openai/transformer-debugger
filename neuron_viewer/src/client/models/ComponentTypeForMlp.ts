// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * The type of component / fundamental unit to use for MLP layers.
 *
 * This determines which types of node appear in the node table to represent the MLP layers.
 * Neurons are the fundamental unit of MLP layers, but autoencoder latents are more interpretable.
 */
export enum ComponentTypeForMlp {
  NEURON = "neuron",
  AUTOENCODER_LATENT = "autoencoder_latent",
}
