import {
  DerivedScalarType,
  ExistingExplanationsRequest,
  ReadService,
  ActivationLocationType,
  PassType,
} from "../client";
import {
  DimensionalityOfActivations,
  Node,
  NodeType,
  getDimensionalityOfActivations,
} from "../types";
import { getDatasetNameBasedOnNodeType } from "./paths";

import assert from "assert";

export function assertUnreachable(x: never): never {
  // this ensures that getDerivedScalarType causes an error during compilation
  // if the switch has not been updated to handle a newly added NodeType
  throw new Error("Unexpected object: " + x);
}

// The "online" parameter should be true in situations where autoencoders are being run with the
// model in real time, and false in situations where we're reading stored values (e.g. in
// NeuronRecords).
export function getDerivedScalarType(
  nodeType: NodeType,
  online: boolean = false
): DerivedScalarType {
  switch (nodeType) {
    case NodeType.MLP_NEURON:
      return DerivedScalarType.MLP_POST_ACT;
    case NodeType.AUTOENCODER_LATENT:
      return online
        ? DerivedScalarType.ONLINE_AUTOENCODER_LATENT
        : DerivedScalarType.AUTOENCODER_LATENT;
    case NodeType.MLP_AUTOENCODER_LATENT:
      return online
        ? DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT
        : DerivedScalarType.MLP_AUTOENCODER_LATENT;
    case NodeType.ATTENTION_AUTOENCODER_LATENT:
      return online
        ? DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT
        : DerivedScalarType.ATTENTION_AUTOENCODER_LATENT;
    // For DSTs per token pair (e.g. in attention heads), we use the unflattened DST for online
    // requests, and the flattened DST for offline requests. This is because the online requests
    // are made to the activation server, which expects the unflattened DST, and the offline
    // requests are made to the neuron records, which store the flattened DST.
    case NodeType.ATTENTION_HEAD:
      return online
        ? DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM
        : DerivedScalarType.ATTN_WRITE_NORM;
    case NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR:
      return online
        ? DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS
        : DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS;
    case NodeType.LAYER:
      assert(false, "getDerivedScalarType should not be called on a layer node");
      break;
    case NodeType.RESIDUAL_STREAM_CHANNEL:
      return DerivedScalarType.RESID_POST_MLP;
    case NodeType.VOCAB_TOKEN:
      assert(false, "getDerivedScalarType should not be called on a vocab token node");
      break;
    case NodeType.QK_CHANNEL:
      assert(false, "getDerivedScalarType should not be called on a qk channel node");
      break;
    case NodeType.V_CHANNEL:
      assert(false, "getDerivedScalarType should not be called on a v channel node");
      break;
    default:
      return assertUnreachable(nodeType);
  }
}

// For now, activationIndexForWithinLayerGrad is only necessary for ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS
export function getActivationIndexForWithinLayerGrad(activeNode: Node) {
  switch (activeNode.nodeType) {
    case NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR:
      return {
        layerIndex: activeNode.layerIndex,
        activationLocationType: ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
        tensorIndices: ["All", activeNode.nodeIndex] as (number | "All")[],
        passType: PassType.FORWARD,
      };
    default:
      return undefined;
  }
}

export const readExistingExplanations = async (
  activeNode: Node,
  explanationDatasetNames?: string[]
) => {
  const request: ExistingExplanationsRequest = {
    dst: getDerivedScalarType(activeNode.nodeType, /* online= */ false),
    layerIndex: activeNode.layerIndex,
    activationIndex: activeNode.nodeIndex,
    explanationDatasets: explanationDatasetNames || [],
    neuronDataset: explanationDatasetNames
      ? undefined
      : getDatasetNameBasedOnNodeType(activeNode.nodeType),
  };

  return await ReadService.readExistingExplanations(request);
};

export const getNodeIdAndDatasets = (activeNode: Node) => {
  return {
    dst: getDerivedScalarType(activeNode.nodeType, /* online= */ false),
    layerIndex: activeNode.layerIndex,
    activationIndex: activeNode.nodeIndex,
    datasets: [getDatasetNameBasedOnNodeType(activeNode.nodeType)],
  };
};

// TODO: this is really used for any NodeType that is a scalar per token (e.g.
// including autoencoder latents as well as MLP neurons). Should rename eventually.
export const readNeuronRecord = async (activeNode: Node) => {
  assert(
    getDimensionalityOfActivations(activeNode.nodeType) ===
      DimensionalityOfActivations.SCALAR_PER_TOKEN
  );
  return await ReadService.readNeuronRecord(getNodeIdAndDatasets(activeNode));
};

// TODO: this is really used for any NodeType that is a scalar per token pair. Should
// rename eventually.
export const readAttentionHeadRecord = async (activeNode: Node) => {
  assert(
    getDimensionalityOfActivations(activeNode.nodeType) ===
      DimensionalityOfActivations.SCALAR_PER_TOKEN_PAIR
  );
  return await ReadService.readAttentionHeadRecord(getNodeIdAndDatasets(activeNode));
};

export const readNeuronDatasetsMetadata = async () => {
  return await ReadService.readNeuronDatasetsMetadata();
};
