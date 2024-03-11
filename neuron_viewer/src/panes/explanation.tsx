import { useState, useEffect } from "react";
import { PaneProps } from ".";
import { SectionTitle } from "../commonUiComponents";
import { AttributedScoredExplanation } from "../client";
import { readExistingExplanations } from "../requests/readRequests";

const ExplanationDisplay: React.FC<PaneProps> = ({ activeNode }) => {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [scoredExplanations, setScoredExplanations] = useState<
    AttributedScoredExplanation[] | null
  >(null);

  useEffect(() => {
    const loadExplanations = async () => {
      console.log("getting explanations");
      const explanationDatasets = new URLSearchParams(window.location.search)
        .get("explanation_datasets")
        ?.split(",");
      setScoredExplanations(await readExistingExplanations(activeNode, explanationDatasets));
      setIsLoading(false);
    };
    loadExplanations();
  }, [activeNode]);

  console.log("scored explanations are", scoredExplanations);

  return (
    <div className="min-w-0 flex-1">
      <SectionTitle>Model-generated explanations</SectionTitle>
      {isLoading ? (
        <OneExplanation explanation={null} score={null} />
      ) : scoredExplanations!.length === 1 ? (
        <OneExplanation
          explanation={scoredExplanations![0].explanation}
          score={scoredExplanations![0].score}
        />
      ) : (
        <div className="flex justify-center w-full">
          <table className="table-auto border-collapse min-w-max">
            <thead>
              <tr>
                <th className="border px-4 py-2">Dataset Name</th>
                <th className="border px-4 py-2">Explanation Text</th>
                <th className="border px-4 py-2">Score</th>
              </tr>
            </thead>
            <tbody>
              {scoredExplanations!.map((item, index) => (
                <tr key={index}>
                  {/*
                   * Allow enough horizontal space to fit typical dataset names and explanations.
                   * Break long dataset names so they don't overflow horizontally.
                   */}
                  <td className="border px-4 py-2 max-w-md break-all">{item.datasetName}</td>
                  <td className="border px-4 py-2 max-w-lg">{item.explanation}</td>
                  <td className="border px-4 py-2">{item.score && item.score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="h-8"></div>
    </div>
  );
};

interface OneExplanationProps {
  explanation: string | null;
  score?: number | null;
}

const OneExplanation: React.FC<OneExplanationProps> = ({ explanation, score }) => (
  <blockquote className="p-1 px-4 mx-1 my-0">
    <p className="py-1">
      <em>{explanation || "loading..."}</em>
    </p>
    <p className="py-1">score: {score ? score.toFixed(2) : "undefined"}</p>
  </blockquote>
);

export default ExplanationDisplay;
