import TokenHeatmap from "../tokenHeatmap";
import {
  DimensionalityOfActivations,
  Node,
  TokenSequenceAndAttentionScalars,
  TokenSequenceAndScalars,
  getDimensionalityOfActivations,
} from "../types";
import { SentencePaneProps } from ".";
import { derivedAttentionScalars, derivedScalars } from "../requests/inferenceRequests";
import { FetchAndDisplayPane, FetchAndDisplayProps } from "./fetchAndDisplayPane";
import { useCallback } from "react";
import TokenHeatmap2d from "../tokenHeatmap2d";
import {
  getDerivedScalarType,
  getActivationIndexForWithinLayerGrad,
  getNodeIdAndDatasets,
} from "../requests/readRequests";

const ActivationsForPrompt: React.FC<SentencePaneProps> = ({ activeNode, sentence }) => {
  const dimensionalityOfActivations = getDimensionalityOfActivations(activeNode.nodeType);

  // This fetch function is used for scalar-per-token NodeTypes / DSTs.
  const fetchTokenSequence = useCallback(async () => {
    const result = await derivedScalars({
      inferenceRequestSpec: {
        prompt: sentence,
      },
      derivedScalarsRequestSpec: {
        dst: getDerivedScalarType(activeNode.nodeType, /* online= */ true),
        layerIndex: activeNode.layerIndex,
        activationIndex: activeNode.nodeIndex,
        // We want normalized activations, and specifically for those activations to be normalized
        // against the max scalar for the neuron in the top dataset examples in the associated
        // NeuronRecord.
        normalizeActivationsUsingNeuronRecord: getNodeIdAndDatasets(activeNode),
      },
    });
    const tokensAsStrings = result.inferenceAndTokenData.tokensAsStrings;
    return tokensAsStrings.map((token, i) => ({
      token,
      scalar: result.derivedScalarsResponseData.activations[i],
      normalizedScalar: result.derivedScalarsResponseData.normalizedActivations![i],
    }));
  }, [activeNode, sentence]);

  // This fetch function is used for scalar-per-token-pair (i.e. attention) NodeTypes / DSTs.
  const fetchTokenSequenceAndAttentionScalars = useCallback(async () => {
    const result = await derivedAttentionScalars({
      inferenceRequestSpec: {
        prompt: sentence,
        // activationIndexForGrad is undefined except for AUTOENCODER_LATENT_BY_TOKEN_PAIR.
        activationIndexForWithinLayerGrad: getActivationIndexForWithinLayerGrad(activeNode),
      },
      derivedAttentionScalarsRequestSpec: {
        dst: getDerivedScalarType(activeNode.nodeType, /* online= */ true),
        layerIndex: activeNode.layerIndex,
        activationIndex: activeNode.nodeIndex,
        // See the comment above about normalizing activations.
        normalizeActivationsUsingNeuronRecord: getNodeIdAndDatasets(activeNode),
      },
    });
    return result.derivedAttentionScalarsResponseData.tokenAndAttentionScalarsList;
  }, [activeNode, sentence]);

  const displayTokenSequence = useCallback<
    FetchAndDisplayProps<
      SentencePaneProps,
      TokenSequenceAndScalars | TokenSequenceAndAttentionScalars
    >["displayDataFunc"]
  >(
    (scalarData, isLoading, showAll, setShowAll) => {
      if (isLoading) {
        return (
          <div className="flex justify-center items-center h-6">
            <div className="w-8 h-8 border-4 border-gray-300 rounded-full animate-spin"></div>
          </div>
        );
      }
      return (
        <>
          <div className="min-w-0 flex-1">
            {dimensionalityOfActivations === DimensionalityOfActivations.SCALAR_PER_TOKEN && (
              <TokenHeatmap tokenSequence={scalarData as TokenSequenceAndScalars} />
            )}
            {dimensionalityOfActivations === DimensionalityOfActivations.SCALAR_PER_TOKEN_PAIR && (
              <TokenHeatmap2d
                tokenSequenceAndAttentionScalars={scalarData as TokenSequenceAndAttentionScalars}
              />
            )}
          </div>
        </>
      );
    },
    [dimensionalityOfActivations]
  );

  var fetchFunction;
  switch (dimensionalityOfActivations) {
    case DimensionalityOfActivations.SCALAR_PER_TOKEN:
      fetchFunction = fetchTokenSequence;
      break;
    case DimensionalityOfActivations.SCALAR_PER_TOKEN_PAIR:
      fetchFunction = fetchTokenSequenceAndAttentionScalars;
      break;
  }

  return (
    <FetchAndDisplayPane<Node, TokenSequenceAndScalars | TokenSequenceAndAttentionScalars>
      paneProps={activeNode}
      fetchDataFunc={fetchFunction}
      displayDataFunc={displayTokenSequence}
    />
  );
};

export default ActivationsForPrompt;
