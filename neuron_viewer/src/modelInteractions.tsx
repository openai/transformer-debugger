// React component that handles interactions with a subject or explainer model. Two interactions are
// currently supported:
// 1) Getting activations for a particular prompt from the subject model.
// 2) Scoring explanations using an explainer model.

import React, { ChangeEvent, KeyboardEvent } from "react";
import { SectionTitle, defaultSmallButtonClasses } from "./commonUiComponents";

type ModelInteractionsProps = {
  onGetActivationsForPrompt: (value: string) => void;
  // Scoring explanations is currently only possible for neurons and autoencoder latents. We don't
  // show this option for attention heads.
  onScoreExplanation?: (value: string) => void;
};

const GET_ACTIVATIONS_FOR_PROMPT = "Get activations for prompt";
const SCORE_EXPLANATION = "Score explanation";

const ModelInteractions: React.FC<ModelInteractionsProps> = ({
  onGetActivationsForPrompt,
  onScoreExplanation,
}) => {
  const [textboxValue, setTextboxValue] = React.useState<string>("");
  const toolkit = [GET_ACTIVATIONS_FOR_PROMPT];
  if (onScoreExplanation) {
    toolkit.push(SCORE_EXPLANATION);
  }

  const [activeTool, setActiveTool] = React.useState<string | null>(
    toolkit.length === 0 ? null : toolkit[0]
  );
  if (toolkit.length === 0) {
    return null;
  }

  return (
    <>
      <SectionTitle>Interact with the model</SectionTitle>
      <div className="mb-10 flex-row">
        <div className="flex flex-flow">
          {toolkit.map((tool, i) => (
            <div style={{ width: 240 }} key={i}>
              {toolkit.length > 1 ? (
                <button className={defaultSmallButtonClasses} onClick={() => setActiveTool(tool)}>
                  {tool}
                </button>
              ) : (
                <p>{tool}</p> // If there's only one tool, don't make it a button.
              )}
            </div>
          ))}
        </div>
        <div>
          <textarea
            rows={5}
            name="comment"
            id="comment"
            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            placeholder="⌘+Enter to run"
            defaultValue={""}
            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setTextboxValue(e.target.value)}
            onKeyDown={(e: KeyboardEvent) => {
              if (e.key === "Enter" && e.metaKey) {
                if (activeTool === GET_ACTIVATIONS_FOR_PROMPT) {
                  onGetActivationsForPrompt(textboxValue);
                } else if (activeTool === SCORE_EXPLANATION && onScoreExplanation) {
                  onScoreExplanation(textboxValue);
                }
              }
            }}
          />
        </div>
      </div>
    </>
  );
};

export default ModelInteractions;
