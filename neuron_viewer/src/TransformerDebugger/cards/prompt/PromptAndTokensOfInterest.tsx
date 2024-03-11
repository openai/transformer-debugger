// Component that allows the user to input a prompt and to select tokens of interest.

import React, { useEffect, useState } from "react";
import { Textarea, Button } from "@nextui-org/react";
import { ExplanatoryTooltip } from "../../common/ExplanatoryTooltip";
import { MultiTokenInput } from "./MultiTokenInput";
import swap from "./swap.png";
import { PromptInferenceParams } from "../inference_params/inferenceParams";

export const PromptAndTokensOfInterest: React.FC<{
  promptInferenceParams: PromptInferenceParams;
  setPromptInferenceParams: React.Dispatch<React.SetStateAction<PromptInferenceParams | null>>;
  fetchInferenceData: () => void;
  side: "Left" | "Right" | null;
}> = ({ promptInferenceParams, setPromptInferenceParams, fetchInferenceData, side }) => {
  const [userMessages, setUserMessages] = useState<string[]>([]);
  const { prompt, targetTokens, distractorTokens } = promptInferenceParams;

  useEffect(() => {
    let tempUserMessages: string[] = [];
    if (prompt[prompt.length - 1] === " ") {
      tempUserMessages.push("Warning: prompt ends with space");
    }
    targetTokens.forEach((value, index) => {
      if (value[0] !== " ") {
        tempUserMessages.push(`Warning: token "${value}" does not start with space`);
      }
    });
    distractorTokens.forEach((value, index) => {
      if (value[0] !== " ") {
        tempUserMessages.push(`Warning: token "${value}" does not start with space`);
      }
    });
    setUserMessages(tempUserMessages);
  }, [prompt, targetTokens, distractorTokens]);

  const title = (side ? side + " p" : "P") + "rompt and tokens of interest";

  const handleSwapTokens = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.preventDefault();
    setPromptInferenceParams({
      ...promptInferenceParams,
      targetTokens: promptInferenceParams.distractorTokens,
      distractorTokens: promptInferenceParams.targetTokens,
    });
  };

  return (
    <div className="flex flex-col gap-2">
      <h2 className="flex flex-row gap-2 items-center">
        <ExplanatoryTooltip explanation="In TDB, we explore why the model assigns higher probability to the target token than the distractor token when sampling the next token for a specific prompt.">
          <span className="text-xl font-bold">{title}</span>
        </ExplanatoryTooltip>
      </h2>
      <div className="mb-2 block">
        <ExplanatoryTooltip explanation="The initial text sequence preceding the target or distractor token. This should typically begin with <|endoftext|>.">
          <label htmlFor="prompt" className="text-sm font-medium text-gray-700">
            Prompt
          </label>
        </ExplanatoryTooltip>
        <Textarea
          id="prompt"
          value={prompt}
          variant="bordered"
          className="font-mono"
          onValueChange={(value) =>
            setPromptInferenceParams({ ...promptInferenceParams, prompt: value })
          }
        />
      </div>
      <div className="flex flex-col gap-2">
        <div className="flex flex-row items-center">
          <ExplanatoryTooltip explanation="The token that corresponds to positive activation along the direction of interest, usually the token the model assigns the highest probability to.">
            {/* Use maxWidth here and below so the MultiTokenInputs are vertically aligned. */}
            <div className="flex-1" style={{ maxWidth: "150px" }}>
              Target token(s):
            </div>
          </ExplanatoryTooltip>
          <MultiTokenInput
            className="flex-1"
            tokens={targetTokens}
            onChange={(newTokens) =>
              setPromptInferenceParams({
                ...promptInferenceParams,
                targetTokens: newTokens,
              })
            }
          />
        </div>
        <button
          onClick={handleSwapTokens}
          style={{ alignSelf: "flex-start", marginTop: "-20px", marginBottom: "-20px" }}
        >
          <img
            src={swap}
            alt="Swap icon"
            style={{ width: "32px", height: "32px", marginLeft: "40px", zIndex: 1000 }}
          />
        </button>
        <div className="flex flex-row items-center">
          <ExplanatoryTooltip explanation="The token that corresponds to negative activation along the direction of interest, usually a plausible but incorrect token.">
            <div className="flex-1" style={{ maxWidth: "150px" }}>
              Distractor token(s):
            </div>
          </ExplanatoryTooltip>
          <MultiTokenInput
            className="flex-1"
            tokens={distractorTokens}
            onChange={(newTokens) =>
              setPromptInferenceParams({
                ...promptInferenceParams,
                distractorTokens: newTokens,
              })
            }
            allowLengthZero={true}
          />
        </div>
      </div>
      <div>
        {userMessages.map((message, index) => (
          <div key={index}>{message}</div>
        ))}
      </div>
      <div className="flex flex-row gap-2">
        <Button
          onClick={(e) => {
            e.preventDefault();
            fetchInferenceData();
          }}
        >
          Submit
        </Button>
      </div>
    </div>
  );
};
