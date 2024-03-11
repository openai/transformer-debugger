import { Node } from "../types";
import ActivationsForPrompt from "./activationsForPrompt";
import DatasetExamples from "./datasetExamples";
import Explanation from "./explanation";
import LogitLens from "./logitLens";
import ScoreExplanation from "./scoreExplanation";

export const PaneComponents = {
  ActivationsForPrompt,
  DatasetExamples,
  Explanation,
  LogitLens,
  ScoreExplanation,
} as const;

export type PaneComponentType = keyof typeof PaneComponents;

export interface PaneProps {
  activeNode: Node;
}

export interface SentencePaneProps extends PaneProps {
  sentence: string;
}

export interface ExplanationPaneProps extends PaneProps {
  explanation: string;
}
