import type {
  ComponentTypeForMlp,
  ComponentTypeForAttention,
  NodeAblation,
  NodeToTrace,
} from "../../../client";

// Prompt-specific parameters. If there are two prompts, all of these can vary between them.
// (Note that we've temporarily forced ablations to match between prompts; they're stored
// exclusively on the left prompt's params.)
export type PromptInferenceParams = {
  prompt: string;
  targetTokens: string[];
  distractorTokens: string[];
  nodeAblations: NodeAblation[];
  upstreamNodeToTrace: NodeToTrace | null;
  downstreamNodeToTrace: NodeToTrace | null;
};

// Non-prompt-specific parameters. If there are two prompts, these are shared between them.
export type CommonInferenceParams = {
  componentTypeForMlp: ComponentTypeForMlp;
  componentTypeForAttention: ComponentTypeForAttention;
  topAndBottomKForNodeTable: number;
  hideEarlyLayersWhenAblating: boolean;
};
