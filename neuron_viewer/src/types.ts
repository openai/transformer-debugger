import { PaneComponentType } from "./panes";
import { TokenAndScalar, TokenAndAttentionScalars } from "./client";
import { NodeType } from "./client/models/NodeType";
import { assertUnreachable } from "./requests/readRequests";

export { NodeType };

export type PaneData = {
  id?: string;
  type: PaneComponentType;
  sentence?: string;
  explanation?: string;
};

export enum DimensionalityOfActivations {
  SCALAR_PER_TOKEN,
  SCALAR_PER_TOKEN_PAIR,
}

export function getDimensionalityOfActivations(nodeType: NodeType) {
  switch (nodeType) {
    case NodeType.MLP_NEURON:
    case NodeType.AUTOENCODER_LATENT:
    case NodeType.MLP_AUTOENCODER_LATENT:
    case NodeType.ATTENTION_AUTOENCODER_LATENT:
    case NodeType.RESIDUAL_STREAM_CHANNEL:
    case NodeType.QK_CHANNEL:
    case NodeType.V_CHANNEL:
      return DimensionalityOfActivations.SCALAR_PER_TOKEN;
    case NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR:
    case NodeType.ATTENTION_HEAD:
      return DimensionalityOfActivations.SCALAR_PER_TOKEN_PAIR;
    case NodeType.LAYER:
    case NodeType.VOCAB_TOKEN:
      throw new Error(`getDimensionalityOfActivations should not be called on ${nodeType}`);
    default:
      return assertUnreachable(nodeType);
  }
}

export enum TopTokenSupported {
  INPUT,
  OUTPUT,
  BOTH,
  NONE,
}

export function getTopTokenSupported(nodeType: NodeType) {
  // Define if a node supports top input-token or top output-token requests
  switch (nodeType) {
    case NodeType.MLP_NEURON:
    case NodeType.AUTOENCODER_LATENT:
    case NodeType.MLP_AUTOENCODER_LATENT:
    case NodeType.RESIDUAL_STREAM_CHANNEL:
      return TopTokenSupported.BOTH;
    case NodeType.ATTENTION_AUTOENCODER_LATENT:
    case NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR:
      return TopTokenSupported.OUTPUT;
    case NodeType.ATTENTION_HEAD:
    case NodeType.QK_CHANNEL:
    case NodeType.V_CHANNEL:
    case NodeType.LAYER:
    case NodeType.VOCAB_TOKEN:
      return TopTokenSupported.NONE;
    default:
      return assertUnreachable(nodeType);
  }
}

export function stringToNodeType(key: string | undefined): NodeType {
  // string used in the URL, might be different from the DST name
  switch (key) {
    case "attention_head":
      return NodeType.ATTENTION_HEAD;
    case "attn_write_norm":
      return NodeType.ATTENTION_HEAD;
    case "mlp_neuron":
      return NodeType.MLP_NEURON;
    case "autoencoder_latent":
      return NodeType.AUTOENCODER_LATENT;
    case "mlp_autoencoder_latent":
      return NodeType.MLP_AUTOENCODER_LATENT;
    case "attention_autoencoder_latent":
      return NodeType.ATTENTION_AUTOENCODER_LATENT;
    case "autoencoder_latent_by_token_pair":
      return NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR;
    case "layer":
      return NodeType.LAYER;
    case "residual_stream_channel":
      return NodeType.RESIDUAL_STREAM_CHANNEL;
    case "vocab_token":
      return NodeType.VOCAB_TOKEN;
    default:
      throw new Error(`Invalid node type string: ${key}`);
  }
}

export function dstStringToNodeType(dst: string): NodeType {
  switch (dst) {
    case "autoencoder_latent":
      return NodeType.AUTOENCODER_LATENT;
    case "mlp_autoencoder_latent":
      return NodeType.MLP_AUTOENCODER_LATENT;
    case "attention_autoencoder_latent":
      return NodeType.ATTENTION_AUTOENCODER_LATENT;
    case "online_autoencoder_latent":
      return NodeType.AUTOENCODER_LATENT;
    case "mlp_post_act":
      return NodeType.MLP_NEURON;
    case "attn_write_norm":
      return NodeType.ATTENTION_HEAD;
    case "flattened_attn_write_to_latent_summed_over_heads":
      return NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR;
    case "flattened_attn_write_to_latent_summed_over_heads_batched":
      return NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR;
    case "residual":
      return NodeType.RESIDUAL_STREAM_CHANNEL;
    // Flesh out with more as we support more derived scalar types.
    default:
      throw new Error(`Invalid derived scalar type: ${dst}`);
  }
}

export type Node = {
  nodeType: NodeType;
  layerIndex: number;
  nodeIndex: number;
};

export type TokenSequenceAndScalars = TokenAndScalar[];

export type TokenSequenceAndAttentionScalars = TokenAndAttentionScalars[];

export type ScoredExplanation = {
  explanation: string;
  score?: string;
};

// We use a Unicode character that is unlikely to appear in prompts.
export const PROMPTS_SEPARATOR = "߷";
