import { OpenAPI } from "../client";
import { NodeType } from "../types";

export const getQueryParams = () => {
  const urlParams = new URLSearchParams(window.location.search);
  const params: { [key: string]: any } = {};
  for (const [key, value] of urlParams.entries()) {
    params[key] = value;
  }
  return params;
};

export function getDatasetName(): string {
  // Get the current top-level URL.
  const url = new URL(window.location.href);
  // Grab the part matching "/<datasetName>/".
  const match = url.pathname.match(/\/([^/]*)\//);
  return match![1];
}

export function getFirstPartOfDatasetName(): string {
  return getDatasetName().split("_")[0];
}

export function getDatasetNameBasedOnNodeType(nodeType: NodeType): string {
  // This function is used for explainerRequests and readRequests, which should only
  // use one autoencoder name. The autoencoder name is infered from the node type.
  if (
    nodeType === NodeType.AUTOENCODER_LATENT ||
    nodeType === NodeType.MLP_AUTOENCODER_LATENT ||
    nodeType === NodeType.ATTENTION_AUTOENCODER_LATENT ||
    nodeType === NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR
  ) {
    // if there are multiple autoencoders, we need to use the full name to disambiguate
    const parts = getDatasetName().split("_");
    const modelName = parts[0];
    const autoencoderName = parts
      .slice(1)
      .find(
        (part) =>
          (nodeType === NodeType.AUTOENCODER_LATENT && part.includes("")) ||
          (nodeType === NodeType.MLP_AUTOENCODER_LATENT &&
            (part.includes("resid-delta-mlp") || part.includes("mlp-post-act"))) ||
          (nodeType === NodeType.ATTENTION_AUTOENCODER_LATENT &&
            part.includes("resid-delta-attn")) ||
          (nodeType === NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR &&
            part.includes("resid-delta-attn"))
      );
    return `${modelName}_${autoencoderName}`;
  }
  // everything other than autoencoder should only use the first part of the dataset name, before the underscore
  return getFirstPartOfDatasetName();
}

export function getActivationServerUrl(): string {
  if (process.env.NEURON_VIEWER_ACTIVATION_SERVER_URL) {
    return process.env.NEURON_VIEWER_ACTIVATION_SERVER_URL;
  }
  return "http://0.0.0.0:8000";
}

OpenAPI.BASE = getActivationServerUrl();
