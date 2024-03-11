import { MirroredNodeIndex } from "../../client";
import { NodeType, Node } from "../../types";
import _ from "lodash";

export function nodeFromNodeIndex(nodeIndex: MirroredNodeIndex): Node {
  const node: Node = {
    nodeType: nodeIndex.nodeType,
    layerIndex: nodeIndex.layerIndex || 0,
    nodeIndex: nodeIndex.tensorIndices.slice(-1)[0],
  };
  return node;
}

export function namedAttentionHeadIndices(nodeIndex: MirroredNodeIndex) {
  if (nodeIndex.nodeType !== NodeType.ATTENTION_HEAD) {
    throw new Error("Incorrect nodeType for namedAttentionHeadIndices function");
  }
  const [attendedFromTokenIndex, attendedToTokenIndex, attentionHeadIndex] =
    nodeIndex.tensorIndices;
  return { attendedFromTokenIndex, attendedToTokenIndex, attentionHeadIndex };
}

export function namedMlpNeuronIndices(nodeIndex: MirroredNodeIndex) {
  if (nodeIndex.nodeType !== NodeType.MLP_NEURON) {
    throw new Error("Incorrect nodeType for namedMlpNeuronIndices function");
  }
  const [sequenceTokenIndex, neuronIndex] = nodeIndex.tensorIndices;
  return { sequenceTokenIndex, neuronIndex };
}

export function namedAutoencoderLatentIndices(nodeIndex: MirroredNodeIndex) {
  const validNodeTypes = [
    NodeType.AUTOENCODER_LATENT,
    NodeType.MLP_AUTOENCODER_LATENT,
    NodeType.ATTENTION_AUTOENCODER_LATENT,
  ];
  if (!validNodeTypes.includes(nodeIndex.nodeType)) {
    throw new Error("Incorrect nodeType for namedAutoencoderLatentIndices function");
  }
  const [sequenceTokenIndex, latentIndex] = nodeIndex.tensorIndices;
  return { sequenceTokenIndex, latentIndex };
}

export function nodeToStringKey(node: Node): string {
  return `${node.nodeType}.${node.layerIndex}.${node.nodeIndex}`;
}

export const makeNodeName = (nodeIndex: MirroredNodeIndex) => {
  const activationIndex = getActivationIndex(nodeIndex);
  if (nodeIndex.nodeType === NodeType.ATTENTION_HEAD) {
    return `attn_L${nodeIndex.layerIndex}_${activationIndex}`;
  } else if (nodeIndex.nodeType === NodeType.MLP_NEURON) {
    return `mlp_L${nodeIndex.layerIndex}_${activationIndex}`;
  } else if (nodeIndex.nodeType === NodeType.AUTOENCODER_LATENT) {
    return `latent_L${nodeIndex.layerIndex}_${activationIndex}`;
  } else if (nodeIndex.nodeType === NodeType.MLP_AUTOENCODER_LATENT) {
    return `mlp_ae_L${nodeIndex.layerIndex}_${activationIndex}`;
  } else if (nodeIndex.nodeType === NodeType.ATTENTION_AUTOENCODER_LATENT) {
    return `attn_ae_L${nodeIndex.layerIndex}_${activationIndex}`;
  } else if (nodeIndex.nodeType === NodeType.LAYER) {
    return `embedding`;
  } else {
    console.log(`Unknown node type ${nodeIndex.nodeType}`);
    return `${nodeIndex.nodeType}.${nodeIndex.layerIndex}.${activationIndex}`;
  }
};

export function getSequenceTokenIndex(nodeIndex: MirroredNodeIndex): number {
  return nodeIndex.tensorIndices[0];
}

export function getActivationIndex(nodeIndex: MirroredNodeIndex): number {
  return nodeIndex.tensorIndices[nodeIndex.tensorIndices.length - 1];
}

export function getAttendedToSequenceTokenIndex(nodeIndex: MirroredNodeIndex): number | undefined {
  if (nodeIndex.nodeType === NodeType.ATTENTION_HEAD) {
    return nodeIndex.tensorIndices[1];
  } else {
    return undefined;
  }
}

export type JointIndexLookupTable = {
  nodeIndices: MirroredNodeIndex[];
  rightArrayIndices: (number | undefined)[];
  leftArrayIndices: (number | undefined)[];
};

export function joinIndices(
  rightIndices: MirroredNodeIndex[],
  leftIndices: MirroredNodeIndex[]
): JointIndexLookupTable {
  // Use _.isEqual to compare index values
  const nodeIndices = _.uniqWith([...rightIndices, ...leftIndices], _.isEqual);
  let rightArrayIndices: (number | undefined)[] = Array<undefined>(nodeIndices.length);
  for (let i = 0; i < rightIndices.length; i++) {
    const rightIndex = rightIndices[i];
    const index = nodeIndices.findIndex((nodeIndex) => _.isEqual(nodeIndex, rightIndex));
    rightArrayIndices[index] = i;
  }
  let leftArrayIndices: (number | undefined)[] = Array<undefined>(nodeIndices.length);
  for (let i = 0; i < leftIndices.length; i++) {
    const ablatedIndex = leftIndices[i];
    const index = nodeIndices.findIndex((nodeIndex) => _.isEqual(nodeIndex, ablatedIndex));
    leftArrayIndices[index] = i;
  }
  return { nodeIndices, rightArrayIndices, leftArrayIndices };
}
