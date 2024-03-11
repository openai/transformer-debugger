import { ExplainerService } from "../client";
import { Node } from "../types";
import { getDatasetNameBasedOnNodeType } from "./paths";
import { getDerivedScalarType } from "./readRequests";

export const explain = async (activeNode: Node) => {
  return ExplainerService.explainerExplain({
    dst: getDerivedScalarType(activeNode.nodeType),
    layerIndex: activeNode.layerIndex,
    activationIndex: activeNode.nodeIndex,
    datasets: [getDatasetNameBasedOnNodeType(activeNode.nodeType)],
  });
};

export const scoreExplanation = async (
  activeNode: Node,
  explanation: string,
  maxSequences: number = 2
) => {
  return ExplainerService.explainerScore({
    dst: getDerivedScalarType(activeNode.nodeType),
    layerIndex: activeNode.layerIndex,
    activationIndex: activeNode.nodeIndex,
    datasets: [getDatasetNameBasedOnNodeType(activeNode.nodeType)],
    explanation,
    maxSequences,
  });
};
