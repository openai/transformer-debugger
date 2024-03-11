import { ExplanationPaneProps } from ".";
import { ScoredExplanation } from "../types";
import { scoreExplanation } from "../requests/explainerRequests";
import { FetchAndDisplayPane, FetchAndDisplayProps } from "./fetchAndDisplayPane";
import { useCallback } from "react";

const ScoreExplanation: React.FC<ExplanationPaneProps> = ({ activeNode, explanation }) => {
  const fetchExplanationScore = useCallback(async () => {
    const score = (await scoreExplanation(activeNode, explanation)).score.toFixed(2);
    return {
      explanation,
      score,
    };
  }, [activeNode, explanation]);

  const displayScoredExplanation = useCallback<
    FetchAndDisplayProps<ExplanationPaneProps, ScoredExplanation>["displayDataFunc"]
  >((scoredExplanation, isLoading) => {
    return (
      <div className="min-w-0 flex-1">
        <div className="text-md text-gray-700" style={{ width: 400 }}>
          <p>
            Scoring explanation
            <span className="inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-md font-medium bg-gray-100 text-gray-800">
              {scoredExplanation.explanation}
            </span>
            score
            <span className="inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-md font-medium bg-gray-100 text-gray-800">
              {isLoading ? "..." : scoredExplanation.score!}
            </span>
          </p>
        </div>
      </div>
    );
  }, []);

  return (
    <FetchAndDisplayPane
      paneProps={activeNode}
      fetchDataFunc={fetchExplanationScore}
      displayDataFunc={displayScoredExplanation}
      initialData={{ explanation }}
    />
  );
};

export default ScoreExplanation;
