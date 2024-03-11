import React from "react";
import { Button, Divider } from "@nextui-org/react";
import { makeNodeName } from "../../utils/nodes";
import { NodeType, InferenceAndTokenData, AttentionTraceType, NodeToTrace } from "../../../client";
import { TokenLabel } from "./TokenLabel";
import { namedAttentionHeadIndices } from "../../utils/nodes";
import { PromptInferenceParams } from "./inferenceParams";

type TraceUpstreamNodeSpecProps = {
  leftInferenceParams: PromptInferenceParams;
  setLeftPromptInferenceParams: React.Dispatch<React.SetStateAction<PromptInferenceParams | null>>;
  inferenceAndTokenData: InferenceAndTokenData | null;
};
export const TraceUpstreamNodeSpec: React.FC<TraceUpstreamNodeSpecProps> = ({
  leftInferenceParams,
  setLeftPromptInferenceParams,
  inferenceAndTokenData,
}) => {
  const upstreamNodeToTrace = leftInferenceParams.upstreamNodeToTrace;
  const downstreamNodeToTrace = leftInferenceParams.downstreamNodeToTrace;
  if (!upstreamNodeToTrace) {
    if (downstreamNodeToTrace) {
      throw new Error("downstreamNodeToTrace should be null if upstreamNodeToTrace is null");
    }
    return null;
  }

  const makeDescription = (
    nodeToTrace: NodeToTrace,
    attendedFromTokenIndex: number,
    attendedToTokenIndex: number | null
  ) => {
    if (nodeToTrace.nodeIndex.nodeType === NodeType.ATTENTION_HEAD) {
      return (
        <div>
          <b>{makeNodeName(nodeToTrace.nodeIndex)}</b> attention from{" "}
          <b>
            <TokenLabel
              index={attendedFromTokenIndex}
              inferenceAndTokenData={inferenceAndTokenData}
            />
          </b>{" "}
          to{" "}
          <b>
            <TokenLabel
              index={attendedToTokenIndex!}
              inferenceAndTokenData={inferenceAndTokenData}
            />
          </b>
        </div>
      );
    } else {
      return (
        <div>
          <b>{makeNodeName(nodeToTrace.nodeIndex)}</b> activation at{" "}
          <b>
            <TokenLabel
              index={attendedFromTokenIndex}
              inferenceAndTokenData={inferenceAndTokenData}
            />
          </b>
        </div>
      );
    }
  };

  let description: JSX.Element | null = null;
  if (upstreamNodeToTrace.nodeIndex.nodeType === NodeType.ATTENTION_HEAD) {
    const { attendedFromTokenIndex, attendedToTokenIndex } = namedAttentionHeadIndices(
      upstreamNodeToTrace.nodeIndex
    );
    let typeDescription = "";
    if (upstreamNodeToTrace.attentionTraceType === AttentionTraceType.Q) {
      typeDescription = "query";
    } else if (upstreamNodeToTrace.attentionTraceType === AttentionTraceType.K) {
      typeDescription = "key";
    } else if (upstreamNodeToTrace.attentionTraceType === AttentionTraceType.V) {
      typeDescription = "value";
    }
    if (typeDescription !== "value" && downstreamNodeToTrace !== null) {
      throw new Error("downstreamNodeToTrace should be null if typeDescription is not 'value'");
    }

    if (typeDescription === "value" && downstreamNodeToTrace === null) {
      description = (
        <div>
          Why did{" "}
          {makeDescription(upstreamNodeToTrace, attendedFromTokenIndex, attendedToTokenIndex)}
          increase the loss? Tracing through <b>{typeDescription}</b>.
        </div>
      );
    } else if (typeDescription === "value" && downstreamNodeToTrace !== null) {
      const downstreamIndices = namedAttentionHeadIndices(downstreamNodeToTrace.nodeIndex);
      const downstreamAttendedFromTokenIndex = downstreamIndices.attendedFromTokenIndex;
      const downstreamAttendedToTokenIndex = downstreamIndices.attendedToTokenIndex;
      description = (
        <div>
          Why did{" "}
          {makeDescription(upstreamNodeToTrace, attendedFromTokenIndex, attendedToTokenIndex)}
          increase{" "}
          {makeDescription(
            downstreamNodeToTrace,
            downstreamAttendedFromTokenIndex,
            downstreamAttendedToTokenIndex
          )}
          ? Tracing through <b>{typeDescription}</b>.
        </div>
      );
    } else {
      description = (
        <div>
          <div>
            What caused{" "}
            {makeDescription(upstreamNodeToTrace, attendedFromTokenIndex, attendedToTokenIndex)}?
          </div>
          <div>
            Tracing through <b>{typeDescription}</b>.
          </div>
        </div>
      );
    }
  } else if (
    upstreamNodeToTrace.nodeIndex.nodeType === NodeType.MLP_NEURON ||
    upstreamNodeToTrace.nodeIndex.nodeType === NodeType.AUTOENCODER_LATENT ||
    upstreamNodeToTrace.nodeIndex.nodeType === NodeType.MLP_AUTOENCODER_LATENT ||
    upstreamNodeToTrace.nodeIndex.nodeType === NodeType.ATTENTION_AUTOENCODER_LATENT
  ) {
    if (downstreamNodeToTrace !== null) {
      throw new Error(
        "downstreamNodeToTrace should be null if nodeIndex is an MLP neuron or latent"
      );
    }
    const indices = upstreamNodeToTrace.nodeIndex.tensorIndices;
    description = (
      // null is passed as the attendedToTokenIndex because it doesn't apply to MLP neurons or latent nodes
      <div>What caused {makeDescription(upstreamNodeToTrace, indices[0], null)}?</div>
    );
  }

  return (
    <div>
      <div className="flex flex-row gap-2 items-center">
        <span className="text-xl font-bold">Trace</span>
      </div>
      <div>{description}</div>
      <div>
        <Button
          onClick={(e) => {
            e.preventDefault();
            setLeftPromptInferenceParams({
              ...leftInferenceParams,
              upstreamNodeToTrace: null,
              downstreamNodeToTrace: null,
            });
          }}
        >
          Clear
        </Button>
      </div>
      <Divider className="my-4" />
    </div>
  );
};
