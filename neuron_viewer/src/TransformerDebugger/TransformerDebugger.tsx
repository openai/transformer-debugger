//  Transformer Debugger, interpretability tool to allow inspecting model activations

import React, { useState, useMemo, useEffect } from "react";
import {
  type MultipleTopKDerivedScalarsResponseData,
  type ModelInfoResponse,
  InferenceResponseAndResponseDict,
  TdbRequestSpec,
} from "../client";
import { useLocation, useNavigate } from "react-router-dom";
import { LogitsDisplay } from "./cards/LogitsDisplay";
import { NodeTable } from "./cards/node_table/NodeTable";
import { InferenceParamsDisplay } from "./cards/inference_params/InferenceParamsDisplay";
import { BySequenceTokenDisplay } from "./cards/BySequenceTokenDisplay";
import { getInferenceAndTokenData, getSubResponse } from "./requests/inferenceResponseUtils";
import { Card, CardBody } from "@nextui-org/react";
import { Link } from "@nextui-org/react";
import { queryToInferenceParams, updateQueryFromInferenceParams } from "./utils/urlParams";
import { useExplanationFetcher } from "./requests/explanationFetcher";
import { InferenceDataFetcher, fetchModelInfo } from "./requests/inferenceDataFetcher";
import DisplayOptions from "./cards/DisplayOptions";
import JsonModal from "./common/JsonModal";
import TokenTable from "./cards/TokenTable";
import {
  CommonInferenceParams,
  PromptInferenceParams,
} from "./cards/inference_params/inferenceParams";

const TransformerDebugger: React.FC = () => {
  // Top level component, should manage all state and pass it down to children
  const location = useLocation();
  const navigate = useNavigate();
  const query = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const {
    commonParams: commonParamsFromUrl,
    leftPromptParams: leftPromptParamsFromUrl,
    rightPromptParams: rightPromptParamsFromUrl,
  } = queryToInferenceParams(query);
  const [commonInferenceParams, setCommonInferenceParams] =
    useState<CommonInferenceParams>(commonParamsFromUrl);
  const [leftPromptInferenceParams, setLeftPromptInferenceParams] =
    useState<PromptInferenceParams | null>(leftPromptParamsFromUrl);
  const [rightPromptInferenceParams, setRightPromptInferenceParams] =
    useState<PromptInferenceParams | null>(rightPromptParamsFromUrl);
  const [twoPromptsMode, setTwoPromptsMode] = React.useState(rightPromptInferenceParams !== null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  if (!leftPromptInferenceParams) {
    throw new Error("leftPromptInferenceParams should never be null");
  }

  const { explanationMap, setNodesRequestingExplanation } = useExplanationFetcher();

  useEffect(() => {
    const updatedQuery = updateQueryFromInferenceParams(
      query,
      commonInferenceParams,
      leftPromptInferenceParams,
      rightPromptInferenceParams
    );
    navigate({ search: updatedQuery.toString() });
  }, [
    commonInferenceParams,
    leftPromptInferenceParams,
    rightPromptInferenceParams,
    navigate,
    query,
  ]);

  // TDB has a concept of left and right requests and responses. In cases where one request is a
  // "test" request and the other is a baseline, the left request is the test request and the right
  // request is the baseline request. In cases where there's only one request, it's the left
  // request. In other cases, left vs. right is arbitrary.
  const [rightRequest, setRightRequest] = useState<TdbRequestSpec | null>(null);
  const [rightResponse, setRightResponse] = useState<InferenceResponseAndResponseDict | null>(null);
  const [leftRequest, setLeftRequest] = useState<TdbRequestSpec | null>(null);
  const [leftResponse, setLeftResponse] = useState<InferenceResponseAndResponseDict | null>(null);
  const [activationServerErrorMessage, setActivationServerErrorMessage] = useState<string | null>(
    null
  );

  const inferenceDataFetcher = new InferenceDataFetcher();
  const fetchInferenceData = React.useCallback(async () => {
    inferenceDataFetcher.fetch(
      modelInfo,
      commonInferenceParams,
      leftPromptInferenceParams,
      rightPromptInferenceParams,
      setRightResponse,
      setLeftResponse,
      setRightRequest,
      setLeftRequest,
      setActivationServerErrorMessage
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commonInferenceParams, leftPromptInferenceParams, rightPromptInferenceParams, modelInfo]);

  useEffect(() => {
    fetchModelInfo(setModelInfo, setActivationServerErrorMessage);
  }, []);

  const prevCommonInferenceParamsRef = React.useRef<CommonInferenceParams>();
  const prevLeftPromptInferenceParamsRef = React.useRef<PromptInferenceParams>();
  const prevRightPromptInferenceParamsRef = React.useRef<PromptInferenceParams | null>();

  // Call fetchInferenceData once on mount and whenever specific inference parameters change.
  useEffect(() => {
    const shouldFetch = inferenceDataFetcher.shouldFetch(
      commonInferenceParams,
      leftPromptInferenceParams,
      rightPromptInferenceParams,
      prevCommonInferenceParamsRef,
      prevLeftPromptInferenceParamsRef,
      prevRightPromptInferenceParamsRef
    );

    if (shouldFetch) {
      fetchInferenceData();
    }

    prevCommonInferenceParamsRef.current = commonInferenceParams;
    prevLeftPromptInferenceParamsRef.current = leftPromptInferenceParams;
    prevRightPromptInferenceParamsRef.current = rightPromptInferenceParams;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    commonInferenceParams,
    leftPromptInferenceParams,
    rightPromptInferenceParams,
    twoPromptsMode,
  ]);

  // Fetch inference data any time the modelInfo changes.
  useEffect(() => {
    fetchInferenceData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelInfo]);

  const [displaySettings, setDisplaySettings] = useState(
    new Map<string, boolean>([
      ["logits", true],
      ["node", true],
      ["bySequenceToken", false],
    ])
  );

  const toggleDisplay = React.useCallback(
    (key: string) => {
      setDisplaySettings((prevSettings) => {
        const newSettings = new Map(prevSettings);
        newSettings.set(key, !newSettings.get(key));
        return newSettings;
      });
    },
    [setDisplaySettings]
  );

  const shouldShowBySequenceTokenDisplay = () => {
    return displaySettings.get("bySequenceToken") && leftResponse;
  };
  const prompts = [leftPromptInferenceParams.prompt];
  if (rightPromptInferenceParams) {
    prompts.push(rightPromptInferenceParams.prompt);
  }

  return (
    <div>
      <div className="flex justify-between m-5">
        <h1 className="text-2xl font-bold">🔭 Transformer Debugger</h1>
        <div className="flex space-x-4">
          <Link
            href="https://github.com/openai/transformer-debugger/blob/main/README.md"
            target="_blank"
            rel="noopener noreferrer"
          >
            Introduction
          </Link>
          <Link
            href="https://github.com/openai/transformer-debugger/blob/main/terminology.md"
            target="_blank"
            rel="noopener noreferrer"
          >
            Terminology
          </Link>
        </div>
      </div>
      <Card>
        <CardBody>
          <InferenceParamsDisplay
            commonInferenceParams={commonInferenceParams}
            setCommonInferenceParams={setCommonInferenceParams}
            leftPromptInferenceParams={leftPromptInferenceParams}
            setLeftPromptInferenceParams={setLeftPromptInferenceParams}
            rightPromptInferenceParams={rightPromptInferenceParams}
            setRightPromptInferenceParams={setRightPromptInferenceParams}
            twoPromptsMode={twoPromptsMode}
            setTwoPromptsMode={setTwoPromptsMode}
            modelInfo={modelInfo}
            fetchInferenceData={fetchInferenceData}
            inferenceAndTokenData={getInferenceAndTokenData(leftResponse)}
          />
        </CardBody>
      </Card>
      <Card>
        <CardBody>
          <div className="flex mt-2 space-x-4">
            <DisplayOptions displaySettings={displaySettings} toggleDisplay={toggleDisplay} />
            <JsonModal
              jsonData={{
                commonInferenceParams,
                leftPromptInferenceParams,
                rightPromptInferenceParams,
                leftRequest,
                rightRequest,
                leftResponse,
                rightResponse,
              }}
            />
          </div>
        </CardBody>
      </Card>
      {activationServerErrorMessage && (
        <Card>
          <CardBody>
            <div className="text-red-500">Error: {activationServerErrorMessage}</div>
          </CardBody>
        </Card>
      )}
      {displaySettings.get("logits") &&
        getSubResponse<MultipleTopKDerivedScalarsResponseData>(
          leftResponse,
          "topOutputTokenLogits"
        ) &&
        getInferenceAndTokenData(leftResponse) && (
          <Card>
            <CardBody>
              <LogitsDisplay
                leftPromptInferenceParams={leftPromptInferenceParams}
                rightPromptInferenceParams={rightPromptInferenceParams}
                rightResponseData={getSubResponse<MultipleTopKDerivedScalarsResponseData>(
                  rightResponse,
                  "topOutputTokenLogits"
                )}
                leftResponseData={
                  getSubResponse<MultipleTopKDerivedScalarsResponseData>(
                    leftResponse,
                    "topOutputTokenLogits"
                  )!
                }
                rightInferenceAndTokenData={getInferenceAndTokenData(rightResponse)}
                leftInferenceAndTokenData={getInferenceAndTokenData(leftResponse)!}
              />
            </CardBody>
          </Card>
        )}
      {shouldShowBySequenceTokenDisplay() && (
        <div className="flex">
          {leftResponse && (
            <div className="w-1/2">
              <Card>
                <CardBody>
                  <BySequenceTokenDisplay
                    responseData={
                      getSubResponse<MultipleTopKDerivedScalarsResponseData>(
                        leftResponse,
                        "componentSumsForTokenDisplay"
                      )!
                    }
                    inferenceAndTokenData={getInferenceAndTokenData(leftResponse)!}
                  />
                </CardBody>
              </Card>
            </div>
          )}
          {rightResponse && (
            <div className="w-1/2">
              <Card>
                <CardBody>
                  <BySequenceTokenDisplay
                    responseData={
                      getSubResponse<MultipleTopKDerivedScalarsResponseData>(
                        rightResponse,
                        "componentSumsForTokenDisplay"
                      )!
                    }
                    inferenceAndTokenData={getInferenceAndTokenData(rightResponse)!}
                  />
                </CardBody>
              </Card>
            </div>
          )}
        </div>
      )}
      {leftResponse && (
        <div className="w-full flex justify-start">
          <Card>
            <CardBody>
              <TokenTable
                leftTokens={getInferenceAndTokenData(leftResponse)!.tokensAsStrings}
                rightTokens={getInferenceAndTokenData(rightResponse)?.tokensAsStrings}
              />
            </CardBody>
          </Card>
        </div>
      )}
      {displaySettings.get("node") &&
        getSubResponse<MultipleTopKDerivedScalarsResponseData>(leftResponse, "topKComponents") &&
        getInferenceAndTokenData(leftResponse) && (
          <Card>
            <CardBody>
              <NodeTable
                leftResponse={leftResponse}
                rightResponse={rightResponse}
                explanationMap={explanationMap}
                setNodesRequestingExplanation={setNodesRequestingExplanation}
                leftPromptInferenceParams={leftPromptInferenceParams}
                setLeftPromptInferenceParams={setLeftPromptInferenceParams}
                prompts={prompts}
                commonInferenceParams={commonInferenceParams}
                setCommonInferenceParams={setCommonInferenceParams}
              />
            </CardBody>
          </Card>
        )}
    </div>
  );
};
export default TransformerDebugger;
