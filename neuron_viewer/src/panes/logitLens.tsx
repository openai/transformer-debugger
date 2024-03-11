// UI component that renders lists of tokens that either up/downvote this component or that are
// up/downvoted by this component, in the spirit of the "logit lens". Some terminology:
//
//  - "Input tokens" are tokens that, when they are the current sequence token, tend to up/downvote
//    this model component (increase/decrease its activation). You can think of this as an operation
//    involving the embedding weights and the model component's input weights.
//  - "Output tokens" are tokens that this model component up/downvotes when it's activated (makes
//    more/less likely in the output logits). You can think of this as an operation involving the
//    model component's output weights and the unembedding weights. This is the classic "logit
//    lens".

import { Tooltip } from "@nextui-org/react";
import { PaneProps } from ".";
import {
  ActivationLocationType,
  DerivedScalarType,
  DerivedScalarsResponseData,
  LossFnName,
  NodeType,
  PassType,
  TokenAndScalar,
} from "../client";
import { SectionTitle, ShowAllOrFewerButton } from "../commonUiComponents";
import { combinedInference } from "../requests/inferenceRequests";
import { getDerivedScalarType } from "../requests/readRequests";
import { formatToken } from "../tokenRendering";
import { FetchAndDisplayPane, FetchAndDisplayProps } from "./fetchAndDisplayPane";
import { useCallback } from "react";
import { TopTokenSupported, getTopTokenSupported } from "../types";

interface LogitLensData {
  inputTokensThatUpvote?: TokenAndScalar[];
  inputTokenThatDownvote?: TokenAndScalar[];
  upvotedOutputTokens?: TokenAndScalar[];
  downvotedOutputTokens?: TokenAndScalar[];
}

function getActivationLocationType(nodeType: NodeType) {
  switch (nodeType) {
    case NodeType.MLP_NEURON:
      return ActivationLocationType.MLP_POST_ACT;
    case NodeType.MLP_AUTOENCODER_LATENT:
      return ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT;
    case NodeType.ATTENTION_AUTOENCODER_LATENT:
    case NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR:
      return ActivationLocationType.ONLINE_ATTENTION_AUTOENCODER_LATENT;
    default:
      throw new Error(`Unsupported node type ${nodeType}`);
  }
}

// Here we do not use the DST that is displayed in the viewer, but the DST that is
// used to calculate the upvoted tokens. This is different for example for
// NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR, which calculates the upvoted tokens
// from the latent activation, ignoring the token pair attribution.
// Note that the autoencoder latent DST needs to be online, because we specify an ablation
// that forces the node to have a positive activation.
function getDstForOutputTokensRequest(nodeType: NodeType) {
  switch (nodeType) {
    case NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR:
      return getDerivedScalarType(NodeType.ATTENTION_AUTOENCODER_LATENT, /* online= */ true);
    default:
      return getDerivedScalarType(nodeType, /* online= */ true);
  }
}

// We use a single-token dummy prompt. One token is enough to get the necessary data.
const DUMMY_PROMPT = "a";
// We never have more than one sub-request spec per sub-request, so we can use a fixed sub-request
// name.
const SUB_REQUEST_NAME = "singleton";

const LogitLens: React.FC<PaneProps> = ({ activeNode }) => {
  const fetchLogitLensData = useCallback(async () => {
    let subRequests = [];

    // Not all node types support both "input" and "output" token requests.
    const topTokenSupported = getTopTokenSupported(activeNode.nodeType);
    if (topTokenSupported === TopTokenSupported.NONE) {
      return null; // Does not display anything
    }

    // Input tokens request:
    if ([TopTokenSupported.INPUT, TopTokenSupported.BOTH].includes(topTokenSupported)) {
      subRequests.push({
        inferenceRequestSpec: {
          // We specify an ablation that allows the DST to calculate values for the appropriate
          // model component, since the layer and activation indices in the request are
          // effectively unused.
          prompt: DUMMY_PROMPT,
          ablationSpecs: [
            {
              index: {
                activationLocationType: getActivationLocationType(activeNode.nodeType),
                tensorIndices: [0, activeNode.nodeIndex], // [token_index, activation_index]
                layerIndex: activeNode.layerIndex,
                passType: PassType.BACKWARD,
              },
              value: 1.0,
            },
          ],
          lossFnConfig: { name: LossFnName.ZERO },
        },
        processingRequestSpecByName: {
          [SUB_REQUEST_NAME]: {
            dst: DerivedScalarType.VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION,
            layerIndex: undefined,
            activationIndex: 0, // Dummy value
            numTopTokens: 100,
          },
        },
      });
    }

    // Output tokens request:
    if ([TopTokenSupported.OUTPUT, TopTokenSupported.BOTH].includes(topTokenSupported)) {
      subRequests.push({
        inferenceRequestSpec: {
          // We specify an ablation that forces the node to have a positive activation, since a
          // value of zero (which can happen with autoencoder latents and with RELU) would prevent
          // us from calculating up/downvoted tokens.
          prompt: DUMMY_PROMPT,
          ablationSpecs: [
            {
              index: {
                activationLocationType: getActivationLocationType(activeNode.nodeType),
                tensorIndices: [0, activeNode.nodeIndex], // [token_index, activation_index]
                layerIndex: activeNode.layerIndex,
                passType: PassType.FORWARD,
              },
              value: 1.0,
            },
          ],
        },
        processingRequestSpecByName: {
          [SUB_REQUEST_NAME]: {
            dst: getDstForOutputTokensRequest(activeNode.nodeType),
            layerIndex: activeNode.layerIndex,
            activationIndex: activeNode.nodeIndex,
            numTopTokens: 100,
          },
        },
      });
    }

    // make the requests
    const batchedResponse = await combinedInference({ inferenceSubRequests: subRequests });
    if (batchedResponse.inferenceSubResponses.length !== subRequests.length) {
      throw new Error("Unexpected number of subresponses");
    }

    let inputTokens, outputTokens;
    let subResponsesIndex = 0;

    // extract the data from the "input" request
    if ([TopTokenSupported.INPUT, TopTokenSupported.BOTH].includes(topTokenSupported)) {
      const inputTokensSubResponse = batchedResponse.inferenceSubResponses[subResponsesIndex];
      subResponsesIndex++;
      const inputTokensDerivedScalarsResponseData =
        inputTokensSubResponse.processingResponseDataByName![
          SUB_REQUEST_NAME
        ] as DerivedScalarsResponseData;
      inputTokens = inputTokensDerivedScalarsResponseData.topTokens;
    }

    // extract the data from the "output" request
    if ([TopTokenSupported.OUTPUT, TopTokenSupported.BOTH].includes(topTokenSupported)) {
      const outputTokensSubResponse = batchedResponse.inferenceSubResponses[subResponsesIndex];
      subResponsesIndex++;
      const outputTokensDerivedScalarsResponseData =
        outputTokensSubResponse.processingResponseDataByName![
          SUB_REQUEST_NAME
        ] as DerivedScalarsResponseData;
      outputTokens = outputTokensDerivedScalarsResponseData.topTokens;
    }

    return {
      inputTokensThatUpvote: inputTokens?.top,
      inputTokenThatDownvote: inputTokens?.bottom,
      upvotedOutputTokens: outputTokens?.top,
      downvotedOutputTokens: outputTokens?.bottom,
    };
  }, [activeNode]);

  const displayLogitLens = useCallback<
    FetchAndDisplayProps<PaneProps, LogitLensData | null>["displayDataFunc"]
  >((logitLensData, isLoading, showAll, setShowAll) => {
    if (logitLensData === null) {
      return <></>; // Does not display anything if given null data
    }

    const maxTokensPerType = showAll ? undefined : 20;
    return (
      <div className="min-w-0 flex-auto">
        <SectionTitle>Logit lens</SectionTitle>
        <div className="flex">
          {/* make two columns, one for the input tokens, one for the output tokens */}
          <div className="w-1/2">
            {logitLensData.inputTokensThatUpvote &&
              renderTokenList(
                "Input tokens that upvote this component",
                logitLensData.inputTokensThatUpvote,
                maxTokensPerType
              )}
            {logitLensData.inputTokenThatDownvote &&
              renderTokenList(
                "Input tokens that downvote this component",
                logitLensData.inputTokenThatDownvote,
                maxTokensPerType,
                "text-red-800"
              )}
          </div>
          <div className="w-1/2">
            {logitLensData.upvotedOutputTokens &&
              renderTokenList(
                "Output tokens upvoted by this component",
                logitLensData.upvotedOutputTokens,
                maxTokensPerType
              )}
            {logitLensData.downvotedOutputTokens &&
              renderTokenList(
                "Output tokens downvoted by this component",
                logitLensData.downvotedOutputTokens,
                maxTokensPerType,
                "text-red-800"
              )}
          </div>
        </div>
        <ShowAllOrFewerButton showAll={showAll} setShowAll={setShowAll} />
        <div className="h-8"></div>
      </div>
    );
  }, []);
  return (
    <FetchAndDisplayPane
      paneProps={activeNode}
      fetchDataFunc={fetchLogitLensData}
      displayDataFunc={displayLogitLens}
    />
  );
};

const renderTokenList = (
  title: string,
  tokensAndScalars: TokenAndScalar[],
  maxTokens?: number,
  customClass: string = ""
) => (
  <div className="mt-2 text-sm text-gray-700">
    <p>
      <Tooltip content="Hover over tokens to see weights, normalized to [-1, 1]">
        <span>{title}:</span>
      </Tooltip>
    </p>
    {tokensAndScalars.slice(0, maxTokens).map((tokenAndScalar, idx) => {
      return (
        <Tooltip content={tokenAndScalar.normalizedScalar.toFixed(2)}>
          <span
            key={idx}
            className={`inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 ${customClass}`}
          >
            {formatToken(tokenAndScalar.token)}
          </span>
        </Tooltip>
      );
    })}
  </div>
);

export default LogitLens;
