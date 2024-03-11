// Display and allow modification of parameters used to build the tdb request.

import React from "react";
import { Switch, Divider, Button } from "@nextui-org/react";
import {
  ModelInfoResponse,
  InferenceAndTokenData,
  ComponentTypeForMlp,
  ComponentTypeForAttention,
} from "../../../client";
import { ExplanatoryTooltip } from "../../common/ExplanatoryTooltip";
import { PromptAndTokensOfInterest } from "../prompt/PromptAndTokensOfInterest";
import { AblateNodeSpecs } from "./AblateNodeSpecs";
import { TraceUpstreamNodeSpec } from "./TraceUpstreamNodeSpec";
import { CommonInferenceParams, PromptInferenceParams } from "./inferenceParams";

export type InferenceParamsDisplayProps = {
  commonInferenceParams: CommonInferenceParams;
  setCommonInferenceParams: React.Dispatch<React.SetStateAction<CommonInferenceParams>>;
  leftPromptInferenceParams: PromptInferenceParams;
  setLeftPromptInferenceParams: React.Dispatch<React.SetStateAction<PromptInferenceParams | null>>;
  rightPromptInferenceParams: PromptInferenceParams | null;
  setRightPromptInferenceParams: React.Dispatch<React.SetStateAction<PromptInferenceParams | null>>;
  twoPromptsMode: boolean;
  setTwoPromptsMode: React.Dispatch<React.SetStateAction<boolean>>;
  modelInfo: ModelInfoResponse | null;
  fetchInferenceData: () => void;
  inferenceAndTokenData: InferenceAndTokenData | null;
};

export const InferenceParamsDisplay: React.FC<InferenceParamsDisplayProps> = ({
  commonInferenceParams,
  setCommonInferenceParams,
  leftPromptInferenceParams,
  setLeftPromptInferenceParams,
  rightPromptInferenceParams,
  setRightPromptInferenceParams,
  twoPromptsMode,
  setTwoPromptsMode,
  modelInfo,
  fetchInferenceData,
  inferenceAndTokenData,
}) => {
  return (
    <form className="flex flex-col gap-2">
      <div className="flex flex-row gap-20">
        <div className="w-1/2">
          <PromptAndTokensOfInterest
            promptInferenceParams={leftPromptInferenceParams}
            setPromptInferenceParams={setLeftPromptInferenceParams}
            fetchInferenceData={fetchInferenceData}
            side={twoPromptsMode ? "Left" : null}
          />
        </div>
        {twoPromptsMode && (
          <div className="w-1/2">
            <div className="flex flex-col gap-2">
              <PromptAndTokensOfInterest
                promptInferenceParams={rightPromptInferenceParams!}
                setPromptInferenceParams={setRightPromptInferenceParams}
                fetchInferenceData={fetchInferenceData}
                side="Right"
              />
              <div className="flex flex-row justify-start">
                <Button
                  onClick={(e) => {
                    e.preventDefault();
                    setRightPromptInferenceParams(null);
                    setTwoPromptsMode(false);
                  }}
                >
                  Remove second prompt
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
      {!twoPromptsMode && (
        <div className="flex flex-row gap-2">
          <Button
            onClick={(e) => {
              e.preventDefault();
              setTwoPromptsMode(true);
              // Copy the left prompt inference params as the initial value for the right.
              setRightPromptInferenceParams(JSON.parse(JSON.stringify(leftPromptInferenceParams)));
            }}
          >
            Add second prompt
          </Button>
        </div>
      )}
      <Divider className="my-4" />
      <div className="flex flex-row gap-2 items-center">
        <TraceUpstreamNodeSpec
          leftInferenceParams={leftPromptInferenceParams}
          setLeftPromptInferenceParams={setLeftPromptInferenceParams}
          inferenceAndTokenData={inferenceAndTokenData}
        />
      </div>
      <div className="flex flex-row gap-2 items-center">
        <AblateNodeSpecs
          leftPromptInferenceParams={leftPromptInferenceParams}
          setLeftPromptInferenceParams={setLeftPromptInferenceParams}
          inferenceAndTokenData={inferenceAndTokenData}
          twoPromptsMode={twoPromptsMode}
        />
      </div>
      <div
        className="flex flex-row gap-10 items-center"
        style={{ marginTop: leftPromptInferenceParams.nodeAblations.length > 0 ? "10px" : "-30px" }}
      >
        <ExplanatoryTooltip explanation="Hides layers before the first ablated layer in the node table.">
          <div>
            <Switch
              isSelected={commonInferenceParams.hideEarlyLayersWhenAblating}
              onValueChange={(unusedValue) => {
                setCommonInferenceParams({
                  ...commonInferenceParams,
                  hideEarlyLayersWhenAblating: !commonInferenceParams.hideEarlyLayersWhenAblating,
                });
              }}
            >
              Hide earlier layers when ablating
            </Switch>
          </div>
        </ExplanatoryTooltip>

        {modelInfo && modelInfo.hasMlpAutoencoder && (
          <ExplanatoryTooltip explanation="Show MLP post-activation autoencoder latents instead of MLP neurons in the node table.">
            <div>
              <Switch
                isSelected={
                  commonInferenceParams.componentTypeForMlp ===
                  ComponentTypeForMlp.AUTOENCODER_LATENT
                }
                onValueChange={(value) => {
                  setCommonInferenceParams({
                    ...commonInferenceParams,
                    componentTypeForMlp: value
                      ? ComponentTypeForMlp.AUTOENCODER_LATENT
                      : ComponentTypeForMlp.NEURON,
                  });
                }}
              >
                Use MLP autoencoders
              </Switch>
            </div>
          </ExplanatoryTooltip>
        )}

        {modelInfo && modelInfo.hasAttentionAutoencoder && (
          <ExplanatoryTooltip explanation="Show attention-write autoencoder latents instead of attention heads in the node table.">
            <div>
              <Switch
                isSelected={
                  commonInferenceParams.componentTypeForAttention ===
                  ComponentTypeForAttention.AUTOENCODER_LATENT
                }
                onValueChange={(value) => {
                  setCommonInferenceParams({
                    ...commonInferenceParams,
                    componentTypeForAttention: value
                      ? ComponentTypeForAttention.AUTOENCODER_LATENT
                      : ComponentTypeForAttention.ATTENTION_HEAD,
                  });
                }}
              >
                Use Attention autoencoders
              </Switch>
            </div>
          </ExplanatoryTooltip>
        )}
      </div>
    </form>
  );
};
