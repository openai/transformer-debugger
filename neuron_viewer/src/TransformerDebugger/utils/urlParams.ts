import { ComponentTypeForMlp, ComponentTypeForAttention } from "../../client";
import {
  CommonInferenceParams,
  PromptInferenceParams,
} from "../cards/inference_params/inferenceParams";

type QueryParamInfo = {
  key: string;
  defaultValue: any;
  // Absent means true.
  jsonEncoded?: boolean;
};

const QueryParams: { [key: string]: QueryParamInfo } = {
  componentTypeForMlp: {
    key: "componentTypeForMlp",
    defaultValue: ComponentTypeForMlp.NEURON,
    jsonEncoded: false,
  },
  componentTypeForAttention: {
    key: "componentTypeForAttention",
    defaultValue: ComponentTypeForAttention.ATTENTION_HEAD,
    jsonEncoded: false,
  },
  topAndBottomKForNodeTable: { key: "topAndBottomKForNodeTable", defaultValue: 10 },
  hideEarlyLayersWhenAblating: {
    key: "hideEarlyLayersWhenAblating",
    defaultValue: true,
  },
  prompt: {
    key: "prompt",
    defaultValue: "<|endoftext|>Paris, France. Ottawa,",
    jsonEncoded: false,
  },
  targetTokens: {
    key: "targetTokens",
    defaultValue: [" Canada"],
  },
  distractorTokens: {
    key: "distractorTokens",
    defaultValue: [" Germany"],
  },
  nodeAblations: { key: "nodeAblations", defaultValue: [] },
  upstreamNodeToTrace: {
    key: "upstreamNodeToTrace",
    defaultValue: null,
  },
  rightPrompt: { key: "rightPrompt", defaultValue: null, jsonEncoded: false },
  downstreamNodeToTrace: {
    key: "downstreamNodeToTrace",
    defaultValue: null,
  },
  rightTargetTokens: {
    key: "rightTargetTokens",
    defaultValue: [],
  },
  rightDistractorTokens: {
    key: "rightDistractorTokens",
    defaultValue: [],
  },
  rightNodeAblations: {
    key: "rightNodeAblations",
    defaultValue: [],
  },
  rightUpstreamNodeToTrace: {
    key: "rightUpstreamNodeToTrace",
    defaultValue: null,
  },
  rightDownstreamNodeToTrace: {
    key: "rightDownstreamNodeToTrace",
    defaultValue: null,
  },
} as const;

function getUrlParamOrDefault(query: URLSearchParams, queryParamInfo: QueryParamInfo) {
  var { key, defaultValue, jsonEncoded } = queryParamInfo;
  // Default to true for jsonEncoded if it's not set.
  jsonEncoded = jsonEncoded ?? true;
  let value = query.get(key);

  if (value === null) {
    return defaultValue;
  } else {
    return jsonEncoded ? JSON.parse(value) : value;
  }
}

function getComponentTypeForMlpFromUrl(query: URLSearchParams): ComponentTypeForMlp {
  const componentTypeForMlpStr = query.get(QueryParams.componentTypeForMlp.key);
  switch (componentTypeForMlpStr) {
    case "neuron":
      return ComponentTypeForMlp.NEURON;
    case "autoencoder": // Legacy value from when the parameter was named "inferenceMode".
    case "autoencoder_latent":
      return ComponentTypeForMlp.AUTOENCODER_LATENT;
    case null:
      return QueryParams.componentTypeForMlp.defaultValue;
    default:
      throw new Error(`Invalid component type for MLP: ${componentTypeForMlpStr}`);
  }
}

function getComponentTypeForAttentionFromUrl(query: URLSearchParams): ComponentTypeForAttention {
  const componentTypeForAttentionStr = query.get(QueryParams.componentTypeForAttention.key);
  switch (componentTypeForAttentionStr) {
    case "attention_head":
      return ComponentTypeForAttention.ATTENTION_HEAD;
    case "autoencoder_latent":
      return ComponentTypeForAttention.AUTOENCODER_LATENT;
    case null:
      return QueryParams.componentTypeForAttention.defaultValue;
    default:
      throw new Error(`Invalid component type for Attention: ${componentTypeForAttentionStr}`);
  }
}

export function queryToInferenceParams(query: URLSearchParams): {
  commonParams: CommonInferenceParams;
  leftPromptParams: PromptInferenceParams;
  rightPromptParams: PromptInferenceParams | null;
} {
  // Common inference params
  const commonParams = {
    componentTypeForMlp: getComponentTypeForMlpFromUrl(query),
    componentTypeForAttention: getComponentTypeForAttentionFromUrl(query),
    topAndBottomKForNodeTable: getUrlParamOrDefault(query, QueryParams.topAndBottomKForNodeTable),
    hideEarlyLayersWhenAblating: getUrlParamOrDefault(
      query,
      QueryParams.hideEarlyLayersWhenAblating
    ),
  };

  // Left prompt params
  const leftPromptParams = {
    prompt: getUrlParamOrDefault(query, QueryParams.prompt),
    targetTokens: getUrlParamOrDefault(query, QueryParams.targetTokens),
    distractorTokens: getUrlParamOrDefault(query, QueryParams.distractorTokens),
    nodeAblations: getUrlParamOrDefault(query, QueryParams.nodeAblations),
    upstreamNodeToTrace: getUrlParamOrDefault(query, QueryParams.upstreamNodeToTrace),
    downstreamNodeToTrace: getUrlParamOrDefault(query, QueryParams.downstreamNodeToTrace),
  };

  // Right prompt params (or null if right prompt is not set)
  let rightPromptParams;
  if (query.has(QueryParams.rightPrompt.key)) {
    rightPromptParams = {
      prompt: getUrlParamOrDefault(query, QueryParams.rightPrompt),
      targetTokens: getUrlParamOrDefault(query, QueryParams.rightTargetTokens),
      distractorTokens: getUrlParamOrDefault(query, QueryParams.rightDistractorTokens),
      nodeAblations: getUrlParamOrDefault(query, QueryParams.rightNodeAblations),
      upstreamNodeToTrace: getUrlParamOrDefault(query, QueryParams.rightUpstreamNodeToTrace),
      downstreamNodeToTrace: getUrlParamOrDefault(query, QueryParams.rightDownstreamNodeToTrace),
    };
  } else {
    rightPromptParams = null;
  }

  return { commonParams, leftPromptParams, rightPromptParams };
}

// If the param's value matches the default, delete it from the query. Otherwise, set it in the
// query. This ensures that only params with non-default values are present in the URL.
export function setOrDelete(query: URLSearchParams, queryParamInfo: QueryParamInfo, value: any) {
  const jsonEncoded = queryParamInfo.jsonEncoded ?? true;
  const valueForUrl = jsonEncoded ? JSON.stringify(value) : value.toString();
  if (String(value) === String(queryParamInfo.defaultValue)) {
    query.delete(queryParamInfo.key);
  } else {
    query.set(queryParamInfo.key, valueForUrl);
  }
}

export function updateQueryFromInferenceParams(
  query: URLSearchParams,
  commonParams: CommonInferenceParams,
  leftPromptParams: PromptInferenceParams,
  rightPromptParams: PromptInferenceParams | null
): URLSearchParams {
  // Common inference params
  setOrDelete(query, QueryParams.componentTypeForMlp, commonParams.componentTypeForMlp);
  setOrDelete(query, QueryParams.componentTypeForAttention, commonParams.componentTypeForAttention);
  setOrDelete(query, QueryParams.topAndBottomKForNodeTable, commonParams.topAndBottomKForNodeTable);
  setOrDelete(
    query,
    QueryParams.hideEarlyLayersWhenAblating,
    commonParams.hideEarlyLayersWhenAblating
  );

  // Left prompt params
  setOrDelete(query, QueryParams.prompt, leftPromptParams.prompt);
  setOrDelete(query, QueryParams.targetTokens, leftPromptParams.targetTokens);
  setOrDelete(query, QueryParams.distractorTokens, leftPromptParams.distractorTokens);
  setOrDelete(query, QueryParams.nodeAblations, leftPromptParams.nodeAblations);
  setOrDelete(query, QueryParams.upstreamNodeToTrace, leftPromptParams.upstreamNodeToTrace);
  setOrDelete(query, QueryParams.downstreamNodeToTrace, leftPromptParams.downstreamNodeToTrace);

  // Right prompt params (if they exist)
  if (rightPromptParams) {
    setOrDelete(query, QueryParams.rightPrompt, rightPromptParams.prompt);
    setOrDelete(query, QueryParams.rightTargetTokens, rightPromptParams.targetTokens);
    setOrDelete(query, QueryParams.rightDistractorTokens, rightPromptParams.distractorTokens);
    setOrDelete(query, QueryParams.rightNodeAblations, rightPromptParams.nodeAblations);
    setOrDelete(query, QueryParams.rightUpstreamNodeToTrace, rightPromptParams.upstreamNodeToTrace);
    setOrDelete(
      query,
      QueryParams.rightDownstreamNodeToTrace,
      rightPromptParams.downstreamNodeToTrace
    );
  } else {
    query.delete(QueryParams.rightPrompt.key);
    query.delete(QueryParams.rightTargetTokens.key);
    query.delete(QueryParams.rightDistractorTokens.key);
    query.delete(QueryParams.rightNodeAblations.key);
    query.delete(QueryParams.rightUpstreamNodeToTrace.key);
    query.delete(QueryParams.rightDownstreamNodeToTrace.key);
  }

  return query;
}
