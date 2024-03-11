import React, { useMemo, useCallback } from "react";
import {
  MultipleTopKDerivedScalarsResponseData,
  InferenceAndTokenData,
  MirroredNodeIndex,
  NodeType,
  TopTokens,
  TopTokensAttendedTo,
  AttentionTraceType,
  NodeAblation,
  GroupId,
  ScoredTokensResponseData,
  InferenceResponseAndResponseDict,
  TokenPairAttributionResponseData,
} from "../../../client";
import { Link } from "react-router-dom";
import { Node, PROMPTS_SEPARATOR } from "../../../types";
import { CommonInferenceParams, PromptInferenceParams } from "../inference_params/inferenceParams";
import { nodeFromNodeIndex, nodeToStringKey } from "../../utils/nodes";
import {
  getSequenceTokenIndex,
  getActivationIndex,
  getAttendedToSequenceTokenIndex,
  makeNodeName,
} from "../../utils/nodes";
import { Dropdown, DropdownTrigger, DropdownMenu, DropdownItem, Button } from "@nextui-org/react";
import {
  ColDef,
  ColGroupDef,
  IsFullWidthRowParams,
  IRowNode,
  RowHeightParams,
  CellStyle,
} from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import { AgGridEvent } from "ag-grid-community";
import "ag-grid-community/styles/ag-grid.css"; // Core grid CSS, always needed
import "ag-grid-community/styles/ag-theme-alpine.css"; // Optional theme CSS
import JsonModal from "../../common/JsonModal";
import { diffOptionalNumbers, formatFloat, compareWithUndefinedAsZero } from "../../utils/numbers";
import {
  ACTIVATION_EXPLANATION,
  ACT_TIMES_GRAD_EXPLANATION,
  DIRECTION_WRITE_EXPLANATION,
  WRITE_MAGNITUDE_EXPLANATION,
  TOKEN_ATTENDED_TO_EXPLANATION,
  TOKEN_ATTRIBUTED_TO_EXPLANATION,
  TOKEN_ATTENDED_FROM_EXPLANATION,
} from "../../utils/explanations";
import { TopTokensDisplay } from "./TopTokensDisplay";
import { ExplanationMap, ExplanationMapEntry } from "../../requests/explanationFetcher";
import { renderTokenOnGray, renderTokenOnBlue } from "../../../tokenRendering";
import { getInferenceAndTokenData, getSubResponse } from "../../requests/inferenceResponseUtils";
import { POSITIVE_NEGATIVE_COLORS, getInterpolatedColor } from "../../../colors";
import { ExplanatoryTooltip } from "../../common/ExplanatoryTooltip";

const METRICS = ["WriteNorm", "DirectionWrite", "ActTimesGrad", "Activation"] as const;
type Metric = typeof METRICS[number];

const GROUP_ID_BY_METRIC: Record<Metric, GroupId> = {
  WriteNorm: GroupId.WRITE_NORM,
  DirectionWrite: GroupId.DIRECTION_WRITE,
  ActTimesGrad: GroupId.ACT_TIMES_GRAD,
  Activation: GroupId.ACTIVATION,
};

type MetricValues = {
  left: number;
  right?: number;
  diff: number;
  // Note that this value if aggregated over *all* nodes, not just the current one.
  maxAbs?: number;
};

export type NodeInfo = {
  nodeIndex: MirroredNodeIndex;
  metrics: Record<Metric, MetricValues>;
  leftAttendedToTokenAsString?: string;
  leftAttendedFromTokenAsString?: string;
  rightAttendedToTokenAsString?: string;
  rightAttendedFromTokenAsString?: string;
  nodeType: NodeType;
  layerIndex: number;
  sequenceTokenIndex: number;
  activationIndex: number;
  attendedToSequenceTokenIndex?: number;
  leftAttributedToSequenceTokenIndex?: number;
  rightAttributedToSequenceTokenIndex?: number;
  tokenIndexOfInterest: number;
  name: string;
  explanationEntry?: ExplanationMapEntry;
  leftTopTokensBySpecName?: Record<TopTokensSpecName, TopTokens | null>;
  rightTopTokensBySpecName?: Record<TopTokensSpecName, TopTokens | null>;
};

// This function returns a partially-initialized NodeInfo object. The rest of the fields are
// populated in useCollatedNodeInfoWithExplanations.
export function createPartialNodeInfo(
  nodeIndex: MirroredNodeIndex,
  tokenIndexOfInterest: number
): NodeInfo {
  return {
    nodeIndex,
    metrics: {} as Record<Metric, MetricValues>,
    nodeType: nodeIndex.nodeType,
    layerIndex: nodeIndex.layerIndex as number,
    sequenceTokenIndex: getSequenceTokenIndex(nodeIndex),
    activationIndex: getActivationIndex(nodeIndex),
    attendedToSequenceTokenIndex: getAttendedToSequenceTokenIndex(nodeIndex),
    tokenIndexOfInterest,
    name: makeNodeName(nodeIndex),
  };
}

function assertNodeIndicesMatchExactly(left: MirroredNodeIndex[], right: MirroredNodeIndex[]) {
  if (left.length !== right.length) {
    throw new Error(
      `Length mismatch: left.length (${left.length}) !== right.length (${right.length})`
    );
  }
  for (let i = 0; i < left.length; i++) {
    const leftAsString = JSON.stringify(left[i]);
    const rightAsString = JSON.stringify(right[i]);
    if (leftAsString !== rightAsString) {
      throw new Error(
        `Mismatched node at index ${i}: left[${i}] (${leftAsString}) !== right[${i}] (${rightAsString})`
      );
    }
  }
}

function useCollatedNodeInfoWithExplanations(
  rightResponseData: MultipleTopKDerivedScalarsResponseData | null,
  leftResponseData: MultipleTopKDerivedScalarsResponseData,
  leftInferenceAndTokenData: InferenceAndTokenData,
  rightInferenceAndTokenData: InferenceAndTokenData | null,
  leftTopTokensBySpecName: Record<TopTokensSpecName, TopTokens[]> | null,
  rightTopTokensBySpecName: Record<TopTokensSpecName, TopTokens[]> | null,
  leftTokenPairAttribution: Array<TopTokensAttendedTo> | null,
  rightTokenPairAttribution: Array<TopTokensAttendedTo> | null,
  explanationMap: ExplanationMap,
  tokenIndexOfInterest: number,
  commonInferenceParams: CommonInferenceParams
) {
  const [collatedNodeInfo] = React.useMemo(() => {
    if (rightResponseData) {
      assertNodeIndicesMatchExactly(leftResponseData.nodeIndices, rightResponseData.nodeIndices);
    }
    const collatedNodeInfo: NodeInfo[] = [];

    const maxAbsByMetric: Record<Metric, number> = {} as Record<Metric, number>;
    for (let i = 0; i < leftResponseData.nodeIndices.length; i++) {
      const nodeIndex = leftResponseData.nodeIndices[i];
      let nodeInfo = createPartialNodeInfo(nodeIndex, tokenIndexOfInterest);

      // For attention-write autoencoder latents, we want to consider them as token-pair nodes.
      // This is used both for the viewer link, and the explanation fetching.
      if (nodeInfo.nodeType === NodeType.ATTENTION_AUTOENCODER_LATENT) {
        // change nodeInfo.nodeType, which is used for the viewer link
        nodeInfo.nodeType = NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR;
        // change nodeIndex.nodeIndex.nodeType, which is used for the explanation fetching (not changing
        // nodeInfo.nodeIndex.nodeType as it creates conflicts elsewhere, but creating a new nodeIndex instead)
        nodeInfo.nodeIndex = {
          ...nodeIndex,
          nodeType: NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR,
        };
      }

      METRICS.forEach((metric) => {
        const leftValue = leftResponseData.activationsByGroupId[GROUP_ID_BY_METRIC[metric]][i];
        const rightValue = rightResponseData?.activationsByGroupId[GROUP_ID_BY_METRIC[metric]][i];
        const diffValue = diffOptionalNumbers(leftValue, rightValue);
        maxAbsByMetric[metric] = Math.max(
          maxAbsByMetric[metric] || 0,
          Math.abs(leftValue ?? 0),
          Math.abs(rightValue ?? 0)
        );

        nodeInfo.metrics[metric] = {
          left: leftValue,
          right: rightValue,
          diff: diffValue,
          // maxAbs is set later
        };
      });
      if (nodeInfo.attendedToSequenceTokenIndex !== null) {
        nodeInfo.leftAttendedToTokenAsString =
          leftInferenceAndTokenData.tokensAsStrings[nodeInfo.attendedToSequenceTokenIndex!];
      }
      if (leftTokenPairAttribution !== null && leftTokenPairAttribution[i] !== null) {
        const tokenAttendedToIndex = leftTokenPairAttribution[i].tokenIndices[0];
        nodeInfo.leftAttributedToSequenceTokenIndex = tokenAttendedToIndex;
        nodeInfo.leftAttendedToTokenAsString =
          leftInferenceAndTokenData.tokensAsStrings[tokenAttendedToIndex];
      }
      nodeInfo.leftAttendedFromTokenAsString =
        leftInferenceAndTokenData.tokensAsStrings[nodeInfo.sequenceTokenIndex];
      if (leftTopTokensBySpecName !== null) {
        nodeInfo.leftTopTokensBySpecName = {};
        TOP_TOKENS_SPEC_NAMES.forEach((specName) => {
          nodeInfo.leftTopTokensBySpecName![specName] = leftTopTokensBySpecName[specName][i];
        });
      }

      if (rightResponseData) {
        if (rightInferenceAndTokenData !== null) {
          if (nodeInfo.attendedToSequenceTokenIndex !== null) {
            nodeInfo.rightAttendedToTokenAsString =
              rightInferenceAndTokenData.tokensAsStrings[nodeInfo.attendedToSequenceTokenIndex!];
          }
          if (rightTokenPairAttribution !== null && rightTokenPairAttribution[i] !== null) {
            const tokenAttendedToIndex = rightTokenPairAttribution[i].tokenIndices[0];
            nodeInfo.rightAttributedToSequenceTokenIndex = tokenAttendedToIndex;
            nodeInfo.rightAttendedToTokenAsString =
              rightInferenceAndTokenData.tokensAsStrings[tokenAttendedToIndex];
          }
          nodeInfo.rightAttendedFromTokenAsString =
            rightInferenceAndTokenData.tokensAsStrings[nodeInfo.sequenceTokenIndex];
        }
        if (rightTopTokensBySpecName !== null) {
          nodeInfo.rightTopTokensBySpecName = {};
          TOP_TOKENS_SPEC_NAMES.forEach((specName) => {
            nodeInfo.rightTopTokensBySpecName![specName] = rightTopTokensBySpecName[specName][i];
          });
        }
      }
      collatedNodeInfo.push(nodeInfo);
    }
    for (let i = 0; i < collatedNodeInfo.length; i++) {
      METRICS.forEach((metric) => {
        collatedNodeInfo[i].metrics[metric].maxAbs = maxAbsByMetric[metric];
      });
    }
    return [collatedNodeInfo];
    // Can't pass top tokens by spec name. Doing so causes the component to re-render infinitely.
    // It's safe to omit it because it only changes when the response data changes, and the response
    // data objects are already included in the dependency array.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    rightResponseData,
    leftResponseData,
    rightInferenceAndTokenData,
    leftInferenceAndTokenData,
    tokenIndexOfInterest,
  ]);

  const collatedNodeInfoWithExplanations = React.useMemo(() => {
    const collatedNodeInfoWithExplanations = collatedNodeInfo.map((nodeInfo) => {
      const key = nodeToStringKey(nodeFromNodeIndex(nodeInfo.nodeIndex));
      const explanationEntry = explanationMap.get(key);
      if (explanationEntry === undefined) {
        return nodeInfo;
      }
      return {
        ...nodeInfo,
        explanationEntry,
      };
    });
    return collatedNodeInfoWithExplanations;
  }, [collatedNodeInfo, explanationMap]);
  return collatedNodeInfoWithExplanations;
}

function isDetailRow(params: IsFullWidthRowParams): boolean {
  return params.rowNode.data.isDetailRow as boolean;
}

function getRowHeight(params: RowHeightParams): number {
  const nodeInfo = params.node.data as NodeInfo;
  if (!params.node.data.isDetailRow) {
    return 25;
  }
  var numUniqueTopTokens = 0;
  for (const specName of TOP_TOKENS_SPEC_NAMES) {
    if (nodeInfo.leftTopTokensBySpecName && nodeInfo.leftTopTokensBySpecName[specName]) {
      numUniqueTopTokens++;
    }
  }
  if (numUniqueTopTokens > 3) {
    throw new Error("Unexpected number of unique top tokens: " + numUniqueTopTokens);
  }
  const numPrompts = nodeInfo.rightTopTokensBySpecName ? 2 : 1;
  return 40 + numPrompts * 25 * numUniqueTopTokens;
}

const TOP_TOKENS_SPEC_NAMES = [
  "upvotedOutputTokens",
  "inputTokensThatUpvoteMlp",
  "inputTokensThatUpvoteAttnQ",
  "inputTokensThatUpvoteAttnK",
];
type TopTokensSpecName = typeof TOP_TOKENS_SPEC_NAMES[number];

const LABELS_BY_SPEC_NAME: Record<TopTokensSpecName, string> = {
  upvotedOutputTokens: "upvoted",
  inputTokensThatUpvoteMlp: "tokens that upvote",
  inputTokensThatUpvoteAttnQ: "tokens that upvote Q",
  inputTokensThatUpvoteAttnK: "tokens that upvote K",
};

const EXPLANATIONS_BY_SPEC_NAME: Record<TopTokensSpecName, { increase: string; decrease: string }> =
  {
    upvotedOutputTokens: {
      increase:
        "Direct effect of the node includes increasing the logits of these tokens. Be aware that when a node's activation is negative, its upvoted and downvoted tokens are flipped relative to when its activation is positive.",
      decrease:
        "Direct effect of the node includes decreasing the logits of these tokens. Be aware that when a node's activation is negative, its upvoted and downvoted tokens are flipped relative to when its activation is positive.",
    },
    inputTokensThatUpvoteMlp: {
      increase: "These tokens make this node more likely to activate",
      decrease: "These tokens make this node less likely to activate",
    },
    inputTokensThatUpvoteAttnQ: {
      increase: "These tokens upvote the attention query associated with this node",
      decrease: "These tokens downvote the attention query associated with this node",
    },
    inputTokensThatUpvoteAttnK: {
      increase: "These tokens upvote the attention key associated with this node",
      decrease: "These tokens downvote the attention key associated with this node",
    },
  };

function getTopTokensBySpecName(
  response: InferenceResponseAndResponseDict | null,
  multiTopKData: MultipleTopKDerivedScalarsResponseData | null
): Record<TopTokensSpecName, TopTokens[]> | null {
  if (response === null) {
    return null;
  }

  const topTokensListBySpecName: Record<string, TopTokens[]> = {};
  for (const specName of TOP_TOKENS_SPEC_NAMES) {
    const scoredTokensResponseData = getSubResponse<ScoredTokensResponseData>(response, specName)!;
    assertNodeIndicesMatchExactly(scoredTokensResponseData.nodeIndices, multiTopKData!.nodeIndices);
    topTokensListBySpecName[specName] = scoredTokensResponseData.topTokensList;
  }
  return topTokensListBySpecName;
}

function getTokenPairAttribution(
  response: InferenceResponseAndResponseDict | null,
  multiTopKData: MultipleTopKDerivedScalarsResponseData | null
): Array<TopTokensAttendedTo> | null {
  if (response === null) {
    return null;
  }
  const specName = "tokenPairAttribution";
  const tokenPairAttributionResponseData = getSubResponse<TokenPairAttributionResponseData>(
    response,
    specName
  )!;
  assertNodeIndicesMatchExactly(
    tokenPairAttributionResponseData.nodeIndices,
    multiTopKData!.nodeIndices
  );
  return tokenPairAttributionResponseData.topTokensAttendedToList;
}

export const NodeTable: React.FC<{
  leftResponse: InferenceResponseAndResponseDict | null;
  rightResponse: InferenceResponseAndResponseDict | null;
  explanationMap: ExplanationMap;
  setNodesRequestingExplanation: (nodes: Node[]) => void;
  leftPromptInferenceParams: PromptInferenceParams;
  setLeftPromptInferenceParams: React.Dispatch<React.SetStateAction<PromptInferenceParams | null>>;
  prompts: string[];
  commonInferenceParams: CommonInferenceParams;
  setCommonInferenceParams: React.Dispatch<React.SetStateAction<CommonInferenceParams>>;
}> = ({
  leftResponse,
  rightResponse,
  explanationMap,
  setNodesRequestingExplanation,
  leftPromptInferenceParams,
  setLeftPromptInferenceParams,
  prompts,
  commonInferenceParams,
  setCommonInferenceParams,
}) => {
  const leftResponseData = getSubResponse<MultipleTopKDerivedScalarsResponseData>(
    leftResponse,
    "topKComponents"
  )!;
  const rightResponseData = getSubResponse<MultipleTopKDerivedScalarsResponseData>(
    rightResponse,
    "topKComponents"
  );
  const leftInferenceAndTokenData = getInferenceAndTokenData(leftResponse)!;
  const rightInferenceAndTokenData = getInferenceAndTokenData(rightResponse);
  const leftTopTokensBySpecName = getTopTokensBySpecName(leftResponse, leftResponseData);
  const rightTopTokensBySpecName = getTopTokensBySpecName(rightResponse, rightResponseData);
  const leftTokenPairAttribution = getTokenPairAttribution(leftResponse, leftResponseData);
  const rightTokenPairAttribution = getTokenPairAttribution(rightResponse, rightResponseData);

  var tokenIndexOfInterest;
  if (leftPromptInferenceParams.upstreamNodeToTrace) {
    // If we're doing a trace, we want to highlight the token we're tracing from.
    tokenIndexOfInterest = leftPromptInferenceParams.upstreamNodeToTrace.nodeIndex.tensorIndices[0];
  } else {
    // If not, we want to highlight the last token in the prompt.
    tokenIndexOfInterest = leftInferenceAndTokenData.tokensAsStrings.length - 1;
  }
  const collatedNodeInfoWithExplanations = useCollatedNodeInfoWithExplanations(
    rightResponseData,
    leftResponseData,
    leftInferenceAndTokenData,
    rightInferenceAndTokenData,
    leftTopTokensBySpecName,
    rightTopTokensBySpecName,
    leftTokenPairAttribution,
    rightTokenPairAttribution,
    explanationMap,
    tokenIndexOfInterest,
    commonInferenceParams
  );
  const makeNodeActionsDropdown = React.useCallback(
    (nodeInfo: NodeInfo, inferenceParams: PromptInferenceParams | null) => {
      // for autoencoder latent by token pair (attention-write autoencoder latents)
      // change to the original node type for ablation and tracing through the node
      const nodeIndex = {
        ...nodeInfo.nodeIndex,
        nodeType:
          nodeInfo.nodeIndex.nodeType === NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR &&
          commonInferenceParams.componentTypeForAttention === "autoencoder_latent"
            ? NodeType.ATTENTION_AUTOENCODER_LATENT
            : nodeInfo.nodeIndex.nodeType,
      };
      const nodeAblation: NodeAblation = {
        nodeIndex: nodeIndex,
        value: 0,
      };
      const upstreamNodeToTrace = {
        nodeIndex: nodeIndex,
      };
      const dropdownItems = [];
      dropdownItems.push(
        <DropdownItem
          key="ablate"
          onClick={(e) => {
            e.preventDefault();
            // Ablations are always applied to both prompts. They're saved exclusively in the left
            // prompt's state.
            setLeftPromptInferenceParams((inferenceParams) => ({
              ...inferenceParams!,
              nodeAblations: [...inferenceParams!.nodeAblations, nodeAblation],
            }));
          }}
        >
          Ablate
        </DropdownItem>
      );
      if (nodeInfo.nodeType !== NodeType.LAYER) {
        if (nodeInfo.nodeType === NodeType.ATTENTION_HEAD) {
          dropdownItems.push(
            <DropdownItem
              key="traceThroughQuery"
              onClick={(e) => {
                e.preventDefault();
                setLeftPromptInferenceParams((inferenceParams) => ({
                  ...inferenceParams!,
                  upstreamNodeToTrace: {
                    ...upstreamNodeToTrace,
                    attentionTraceType: AttentionTraceType.Q,
                  },
                }));
              }}
            >
              Trace through query
            </DropdownItem>
          );
          dropdownItems.push(
            <DropdownItem
              key="traceThroughKey"
              onClick={(e) => {
                e.preventDefault();
                setLeftPromptInferenceParams((inferenceParams) => ({
                  ...inferenceParams!,
                  upstreamNodeToTrace: {
                    ...upstreamNodeToTrace,
                    attentionTraceType: AttentionTraceType.K,
                  },
                }));
              }}
            >
              Trace through key
            </DropdownItem>
          );
          dropdownItems.push(
            <DropdownItem
              key="traceThroughQueryAndKey"
              onClick={(e) => {
                e.preventDefault();
                setLeftPromptInferenceParams((inferenceParams) => ({
                  ...inferenceParams!,
                  upstreamNodeToTrace: {
                    ...upstreamNodeToTrace,
                    attentionTraceType: AttentionTraceType.QK,
                  },
                }));
              }}
            >
              Trace through query and key
            </DropdownItem>
          );
          if (
            !inferenceParams!.upstreamNodeToTrace ||
            inferenceParams!.upstreamNodeToTrace.attentionTraceType !== AttentionTraceType.V
          ) {
            dropdownItems.push(
              <DropdownItem
                key="traceThroughValue"
                onClick={(e) => {
                  e.preventDefault();
                  setLeftPromptInferenceParams((inferenceParams) => ({
                    ...inferenceParams!,
                    upstreamNodeToTrace: {
                      ...upstreamNodeToTrace,
                      attentionTraceType: AttentionTraceType.V,
                    },
                    downstreamNodeToTrace: inferenceParams!.upstreamNodeToTrace,
                  }));
                }}
              >
                Trace through value
              </DropdownItem>
            );
          }
        } else if (
          [
            NodeType.MLP_NEURON,
            NodeType.AUTOENCODER_LATENT,
            NodeType.MLP_AUTOENCODER_LATENT,
            NodeType.ATTENTION_AUTOENCODER_LATENT,
            NodeType.AUTOENCODER_LATENT_BY_TOKEN_PAIR,
          ].includes(nodeInfo.nodeType)
        ) {
          dropdownItems.push(
            <DropdownItem
              key="trace"
              onClick={(e) => {
                e.preventDefault();
                setLeftPromptInferenceParams((inferenceParams) => ({
                  ...inferenceParams!,
                  upstreamNodeToTrace,
                }));
              }}
            >
              Trace
            </DropdownItem>
          );
        }
      }
      return (
        <Dropdown>
          <DropdownTrigger>
            <Button variant="bordered">Actions</Button>
          </DropdownTrigger>
          <DropdownMenu aria-label={`Actions for ${nodeInfo.name}`}>{dropdownItems}</DropdownMenu>
        </Dropdown>
      );
    },
    [setLeftPromptInferenceParams, commonInferenceParams]
  );

  const onChanged = (params: AgGridEvent) => {
    let nodesToRequest: Node[] = [];
    params.api.forEachNodeAfterFilterAndSort((nodeInfo, index) => {
      // Each node has two rows in the table. Only request an explanation for one of
      // those rows to avoid requesting the same explanation twice.
      if (
        index < 30 &&
        nodeInfo.data.explanationEntry === undefined &&
        !nodeInfo.data.isDetailRow
      ) {
        nodesToRequest.push(nodeFromNodeIndex(nodeInfo.data.nodeIndex));
      }
    });
    setNodesRequestingExplanation(nodesToRequest);
  };

  interface TokenCellParams {
    leftTokenAsString?: string;
    rightTokenAsString?: string;
    leftSequenceTokenIndex?: number;
    rightSequenceTokenIndex?: number;
    shouldHighlight?: boolean;
  }

  const TokenCell: React.FC<TokenCellParams> = ({
    leftTokenAsString,
    rightTokenAsString,
    leftSequenceTokenIndex,
    rightSequenceTokenIndex,
    shouldHighlight,
  }) => {
    const renderContent = () => {
      const renderToken = shouldHighlight ? renderTokenOnBlue : renderTokenOnGray;
      if (
        leftTokenAsString &&
        rightTokenAsString &&
        leftTokenAsString !== rightTokenAsString &&
        leftSequenceTokenIndex === rightSequenceTokenIndex
      ) {
        return (
          <>
            {renderToken(leftTokenAsString)} / {renderToken(rightTokenAsString)} (
            {leftSequenceTokenIndex})
          </>
        );
      } else if (
        leftTokenAsString &&
        rightTokenAsString &&
        leftTokenAsString !== rightTokenAsString &&
        leftSequenceTokenIndex !== rightSequenceTokenIndex
      ) {
        return (
          <>
            {renderToken(leftTokenAsString)} ({leftSequenceTokenIndex}) /{" "}
            {renderToken(rightTokenAsString)} ({rightSequenceTokenIndex})
          </>
        );
      } else if (leftTokenAsString) {
        return (
          <>
            {renderToken(leftTokenAsString)} ({leftSequenceTokenIndex})
          </>
        );
      } else if (rightTokenAsString) {
        return (
          <>
            {renderToken(rightTokenAsString)} ({rightSequenceTokenIndex})
          </>
        );
      } else {
        throw new Error("No token string found");
      }
    };

    return <div>{renderContent()}</div>;
  };

  function getMaxAbsValueForColumn(colId: string, nodeInfo: NodeInfo) {
    for (const metric of METRICS) {
      if (colId.startsWith(`metrics.${metric}.`)) {
        return nodeInfo.metrics[metric].maxAbs || 0;
      }
    }
    throw new Error("Unhandled column ID: " + colId);
  }

  const columnDefs: (ColDef<NodeInfo, any> | ColGroupDef<NodeInfo>)[] = useMemo(() => {
    const defaultFloatColDefs: ColDef<NodeInfo, any> = {
      valueFormatter: (params: any) => formatFloat(params.value),
      resizable: true,
      width: 80,
      sortingOrder: ["desc", "asc"],
      comparator: compareWithUndefinedAsZero,
      cellStyle: (params: any): CellStyle => {
        const nodeInfo = params.data as NodeInfo;
        const value = (params.value as number) || 0;
        const color = getInterpolatedColor(
          POSITIVE_NEGATIVE_COLORS,
          [-1, 0, 1],
          value / getMaxAbsValueForColumn(params.column.colId, nodeInfo)
        );
        return { backgroundColor: `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)` };
      },
    };
    let columnDefs: (ColDef<NodeInfo, any> | ColGroupDef<NodeInfo>)[] = [
      {
        headerName: "Name",
        // Ensure that the column never scrolls out of view when scrolling horizontally.
        pinned: "left",
        sortable: true,
        filter: true,
        field: "name",
        width: 140,
        cellRenderer: (params: any) => {
          const nodeInfo = params.data as NodeInfo;
          return (
            <Link
              target="_blank"
              className={
                nodeInfo.name.startsWith("attn")
                  ? "text-green-500 hover:text-green-700" // attention in green
                  : "text-blue-500 hover:text-blue-700" // everything else in blue
              }
              to={`../${nodeInfo.nodeType}/${nodeInfo.layerIndex}/${
                nodeInfo.activationIndex
              }?promptsOfInterest=${prompts.join(PROMPTS_SEPARATOR)}`}
              relative="path"
            >
              {nodeInfo.name}
            </Link>
          );
        },
      },
      {
        headerName: "Tokens",
        children: [
          {
            headerName:
              commonInferenceParams.componentTypeForAttention === "autoencoder_latent"
                ? "Attributed to"
                : "Attended to",
            headerTooltip:
              commonInferenceParams.componentTypeForAttention === "autoencoder_latent"
                ? TOKEN_ATTRIBUTED_TO_EXPLANATION
                : TOKEN_ATTENDED_TO_EXPLANATION,
            field: "attendedToSequenceTokenIndex",
            width: 150,
            cellRenderer: (params: any) => {
              const nodeInfo = params.data as NodeInfo;
              // for attention heads, split by token pairs, both left and right attend to the same token index
              if (nodeInfo.attendedToSequenceTokenIndex !== undefined) {
                return (
                  <TokenCell
                    leftTokenAsString={nodeInfo.leftAttendedToTokenAsString}
                    rightTokenAsString={nodeInfo.rightAttendedToTokenAsString}
                    leftSequenceTokenIndex={nodeInfo.attendedToSequenceTokenIndex}
                    rightSequenceTokenIndex={nodeInfo.attendedToSequenceTokenIndex}
                  />
                );
              }
              // for attention latents, left and right might have different attribution for top token attended to
              else if (
                nodeInfo.leftAttributedToSequenceTokenIndex !== undefined ||
                nodeInfo.rightAttributedToSequenceTokenIndex !== undefined
              ) {
                return (
                  <TokenCell
                    leftTokenAsString={nodeInfo.leftAttendedToTokenAsString}
                    rightTokenAsString={nodeInfo.rightAttendedToTokenAsString}
                    leftSequenceTokenIndex={nodeInfo.leftAttributedToSequenceTokenIndex}
                    rightSequenceTokenIndex={nodeInfo.rightAttributedToSequenceTokenIndex}
                  />
                );
              } else {
                return "";
              }
            },
          },
          {
            headerName: "Attended from",
            headerTooltip: TOKEN_ATTENDED_FROM_EXPLANATION,
            field: "sequenceTokenIndex",
            width: 150,
            cellRenderer: (params: any) => {
              const nodeInfo = params.data as NodeInfo;
              return (
                <TokenCell
                  leftTokenAsString={nodeInfo.leftAttendedFromTokenAsString}
                  rightTokenAsString={nodeInfo.rightAttendedFromTokenAsString}
                  leftSequenceTokenIndex={nodeInfo.sequenceTokenIndex}
                  rightSequenceTokenIndex={nodeInfo.sequenceTokenIndex}
                  shouldHighlight={nodeInfo.sequenceTokenIndex === nodeInfo.tokenIndexOfInterest}
                />
              );
            },
          },
        ],
      },
    ];
    if (rightResponseData !== null) {
      columnDefs.push({
        headerName: "Activation",
        headerTooltip: ACTIVATION_EXPLANATION,
        minWidth: 400,
        children: [
          {
            ...defaultFloatColDefs,
            headerName: "Left",
            field: "metrics.Activation.left",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Diff",
            field: "metrics.Activation.diff",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Right",
            field: "metrics.Activation.right",
          },
        ],
      });
    } else {
      columnDefs.push({
        headerName: "Activation",
        headerTooltip: ACTIVATION_EXPLANATION,
        ...defaultFloatColDefs,
        width: 100,
        sortable: true,
        field: "metrics.Activation.left",
      });
    }
    if (rightResponseData !== null) {
      columnDefs.push({
        headerName: "Write magnitude",
        headerTooltip: WRITE_MAGNITUDE_EXPLANATION,
        minWidth: 400,
        children: [
          {
            ...defaultFloatColDefs,
            headerName: "Left",
            field: "metrics.WriteNorm.left",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Diff",
            field: "metrics.WriteNorm.diff",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Right",
            field: "metrics.WriteNorm.right",
          },
        ],
      });
    } else {
      columnDefs.push({
        headerName: "Write magnitude",
        headerTooltip: WRITE_MAGNITUDE_EXPLANATION,
        ...defaultFloatColDefs,
        width: 100,
        sortable: true,
        field: "metrics.WriteNorm.left",
      });
    }
    if (rightResponseData !== null) {
      columnDefs.push({
        headerName: "Direct effect",
        headerTooltip: DIRECTION_WRITE_EXPLANATION,
        children: [
          {
            ...defaultFloatColDefs,
            headerName: "Left",
            field: "metrics.DirectionWrite.left",
            initialSort: "desc",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Diff",
            field: "metrics.DirectionWrite.diff",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Right",
            field: "metrics.DirectionWrite.right",
          },
        ],
      });
    } else {
      columnDefs.push({
        headerName: "Direct effect",
        headerTooltip: DIRECTION_WRITE_EXPLANATION,
        ...defaultFloatColDefs,
        width: 100,
        sortable: true,
        field: "metrics.DirectionWrite.left",
        initialSort: "desc",
      });
    }
    if (rightResponseData !== null) {
      columnDefs.push({
        headerName: "Estimated total effect",
        headerTooltip: ACT_TIMES_GRAD_EXPLANATION,
        children: [
          {
            ...defaultFloatColDefs,
            headerName: "Left",
            field: "metrics.ActTimesGrad.left",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Diff",
            field: "metrics.ActTimesGrad.diff",
          },
          {
            ...defaultFloatColDefs,
            headerName: "Right",
            field: "metrics.ActTimesGrad.right",
          },
        ],
      });
    } else {
      columnDefs.push({
        ...defaultFloatColDefs,
        headerName: "Estimated total effect",
        headerTooltip: ACT_TIMES_GRAD_EXPLANATION,
        field: "metrics.ActTimesGrad.left",
        width: 150,
        sortable: true,
      });
    }
    return columnDefs;
  }, [rightResponseData, prompts, commonInferenceParams.componentTypeForAttention]);

  // To allow each node to cover two rows in AG Grid, we duplicate each row and set the
  // isDetailRow property to true on the second row. The detail rows will use the detailRowRenderer.
  // Sorting works because the original data has the normal row always first.
  const rowsForGrid = React.useMemo(() => {
    return collatedNodeInfoWithExplanations.flatMap((row) => [
      { ...row, isDetailRow: false },
      { ...row, isDetailRow: true },
    ]);
  }, [collatedNodeInfoWithExplanations]);

  const detailRowRenderer = useCallback(
    (params: IRowNode) => {
      const row: NodeInfo = params.data as NodeInfo;
      let explanationText = "";
      const explanationEntry = row.explanationEntry;
      if (explanationEntry === undefined) {
        explanationText = "";
      } else {
        if (explanationEntry.state === "in_progress") {
          explanationText = "Loading...";
        } else if (explanationEntry.state === "error") {
          explanationText = "Error";
        } else {
          const explanationListForCurrentNode = explanationEntry.scoredExplanations;
          explanationText = explanationListForCurrentNode!
            .map((explanation, index) => {
              const scoreStr = explanation.score ? explanation.score.toFixed(2) : "?";
              return `${scoreStr}: ${explanation.explanation}`;
            })
            .join("\n");
        }
      }
      // Nodes other than attention heads will always have the same tokens on both sides, so we
      // only show one set of tokens
      return (
        <div>
          <div>
            {makeNodeActionsDropdown(params.data, leftPromptInferenceParams)} {explanationText}
          </div>
          {Object.keys(row.leftTopTokensBySpecName!).map((specName) => {
            if (row.leftTopTokensBySpecName![specName]) {
              return (
                <div>
                  <TopTokensDisplay
                    leftSideData={row.leftTopTokensBySpecName![specName]}
                    rightSideData={
                      row.rightTopTokensBySpecName ? row.rightTopTokensBySpecName[specName] : null
                    }
                    label={LABELS_BY_SPEC_NAME[specName]}
                    explanations={EXPLANATIONS_BY_SPEC_NAME[specName]}
                  />
                </div>
              );
            }
            return null;
          })}
        </div>
      );
    },
    [makeNodeActionsDropdown, leftPromptInferenceParams]
  );

  return (
    <div>
      <div className="flex justify-between">
        <h2 className="text-xl">Node table</h2>
        <div>
          <label>
            <ExplanatoryTooltip
              explanation={
                "The node table calculates the top k items for each column " +
                "(and bottom k for some columns). When sorting by a column, " +
                "items beyond the first k may no longer be 'top': they will " +
                "depend on which items were in the top k for other columns. " +
                "Adjust k here to control how many top/bottom items are " +
                "calculated for each column. Larger values mean slower performance."
              }
            >
              <span>Top k per column:&nbsp;</span>
            </ExplanatoryTooltip>
            <input
              className="border border-gray-400 w-8 text-right"
              defaultValue={commonInferenceParams.topAndBottomKForNodeTable}
              onBlur={(e) =>
                setCommonInferenceParams({
                  ...commonInferenceParams,
                  topAndBottomKForNodeTable: parseInt(e.target.value),
                })
              }
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  setCommonInferenceParams({
                    ...commonInferenceParams,
                    topAndBottomKForNodeTable: parseInt(e.currentTarget.value),
                  });
                }
              }}
            />
          </label>
        </div>
        <JsonModal
          jsonData={{
            leftResponseData,
            leftInferenceAndTokenData,
            rightResponseData,
            rightInferenceAndTokenData,
            collatedNodeInfoWithExplanations,
          }}
        />
      </div>

      <div
        className="ag-theme-alpine"
        style={{ height: 600, width: "100%", "--ag-grid-size": "3px" } as React.CSSProperties}
      >
        <AgGridReact
          key={"grid"}
          columnDefs={columnDefs}
          rowData={rowsForGrid}
          pagination={true}
          paginationPageSize={30}
          // Used to allow detail rows to render as one cell spanning all columns
          isFullWidthRow={isDetailRow}
          fullWidthCellRenderer={detailRowRenderer}
          defaultColDef={{
            resizable: true,
            autoHeaderHeight: true,
            wrapHeaderText: true,
            sortable: true,
            filter: true,
            filterParams: { newRowsAction: "keep" },
            autoHeight: true,
          }}
          getRowId={
            // Must be unique per row, so we include the isDetailRow value.
            (params) =>
              JSON.stringify(params.data.nodeIndex) +
              " " +
              (params.data.isDetailRow ? "detail" : "")
          }
          onSortChanged={onChanged}
          onRowDataUpdated={onChanged}
          onFilterChanged={onChanged}
          getRowHeight={getRowHeight}
        />
      </div>
    </div>
  );
};
