import {
  ApiError,
  InferenceResponseAndResponseDict,
  ModelInfoResponse,
  NodeAblation,
  OpenAPI,
  TdbRequestSpec,
} from "../../client";
import { batchedTdb, getModelInfo } from "../../requests/inferenceRequests";
import {
  PromptInferenceParams,
  CommonInferenceParams,
} from "../cards/inference_params/inferenceParams";

export const fetchModelInfo = (
  setModelInfo: (info: ModelInfoResponse | null) => void,
  setError: (errorMessage: string | null) => void
) => {
  getModelInfo()
    .then((modelInfo) => {
      console.log("Got model info", modelInfo);
      setModelInfo(modelInfo);
    })
    .catch((_) => {
      const errorMessage =
        "Unable to look up model info. Are you sure you're running an activation server for this dataset? Current URL: " +
        OpenAPI.BASE;
      console.log(errorMessage);
      setError(errorMessage);
    });
};

const buildTdbRequestSpec = (
  {
    prompt,
    targetTokens,
    distractorTokens,
    nodeAblations,
    upstreamNodeToTrace,
  }: PromptInferenceParams,
  {
    componentTypeForMlp,
    componentTypeForAttention,
    topAndBottomKForNodeTable,
    hideEarlyLayersWhenAblating,
  }: CommonInferenceParams
): TdbRequestSpec => {
  return {
    prompt,
    targetTokens,
    distractorTokens,
    componentTypeForMlp,
    componentTypeForAttention,
    topAndBottomKForNodeTable,
    hideEarlyLayersWhenAblating,
    nodeAblations,
    upstreamNodeToTrace: upstreamNodeToTrace === null ? undefined : upstreamNodeToTrace,
  };
};

export class InferenceDataFetcher {
  shouldFetch(
    commonInferenceParams: CommonInferenceParams,
    leftPromptInferenceParams: PromptInferenceParams,
    rightPromptInferenceParams: PromptInferenceParams | null,
    prevCommonInferenceParamsRef: React.MutableRefObject<CommonInferenceParams | undefined>,
    prevLeftPromptInferenceParamsRef: React.MutableRefObject<PromptInferenceParams | undefined>,
    prevRightPromptInferenceParamsRef: React.MutableRefObject<
      PromptInferenceParams | null | undefined
    >
  ) {
    const prevCommonInferenceParams = prevCommonInferenceParamsRef.current;
    const prevLeftPromptInferenceParams = prevLeftPromptInferenceParamsRef.current;
    const prevRightPromptInferenceParams = prevRightPromptInferenceParamsRef.current;
    if (
      prevCommonInferenceParams === undefined ||
      prevLeftPromptInferenceParams === undefined ||
      prevRightPromptInferenceParams === undefined
    ) {
      console.log("Fetching data because previous inference parameters were undefined");
      return true;
    } else {
      if (
        prevCommonInferenceParams.componentTypeForMlp !== commonInferenceParams.componentTypeForMlp
      ) {
        console.log("Fetching data because componentTypeForMlp changed");
        return true;
      }
      if (
        prevCommonInferenceParams.componentTypeForAttention !==
        commonInferenceParams.componentTypeForAttention
      ) {
        console.log("Fetching data because componentTypeForAttention changed");
        return true;
      }
      if (
        prevCommonInferenceParams.topAndBottomKForNodeTable !==
        commonInferenceParams.topAndBottomKForNodeTable
      ) {
        console.log("Fetching data because topAndBottomKForNodeTable changed");
        return true;
      }
      if (prevLeftPromptInferenceParams.nodeAblations !== leftPromptInferenceParams.nodeAblations) {
        console.log("Fetching data because leftPrompt nodeAblations changed");
        return true;
      }
      if (
        prevLeftPromptInferenceParams.upstreamNodeToTrace !==
        leftPromptInferenceParams.upstreamNodeToTrace
      ) {
        console.log("Fetching data because leftPrompt upstreamNodeToTrace changed");
        return true;
      }
      if (prevRightPromptInferenceParams && rightPromptInferenceParams !== null) {
        if (
          prevRightPromptInferenceParams.nodeAblations !== rightPromptInferenceParams.nodeAblations
        ) {
          console.log("Fetching data because rightPrompt nodeAblations changed");
          return true;
        }
        if (
          prevRightPromptInferenceParams.upstreamNodeToTrace !==
          rightPromptInferenceParams.upstreamNodeToTrace
        ) {
          console.log("Fetching data because rightPrompt upstreamNodeToTrace changed");
          return true;
        }
      }
      if (!prevRightPromptInferenceParams && rightPromptInferenceParams) {
        console.log("Fetching data because rightPrompt was added");
        return true;
      }
      if (prevRightPromptInferenceParams && rightPromptInferenceParams === null) {
        console.log("Fetching data because rightPrompt was removed");
        return true;
      }
    }
    return false;
  }

  async fetch(
    modelInfo: ModelInfoResponse | null,
    commonInferenceParams: CommonInferenceParams,
    leftPromptInferenceParams: PromptInferenceParams,
    rightPromptInferenceParams: PromptInferenceParams | null,
    setRightResponse: React.Dispatch<React.SetStateAction<InferenceResponseAndResponseDict | null>>,
    setLeftResponse: React.Dispatch<React.SetStateAction<InferenceResponseAndResponseDict | null>>,
    setRightRequest: React.Dispatch<React.SetStateAction<TdbRequestSpec | null>>,
    setLeftRequest: React.Dispatch<React.SetStateAction<TdbRequestSpec | null>>,
    setActivationServerErrorMessage: React.Dispatch<React.SetStateAction<string | null>>
  ) {
    if (modelInfo === null) {
      return;
    }
    setRightResponse(null);
    setLeftResponse(null);
    const handleInferenceError = this.handleInferenceError;
    function performInference(
      subRequests: TdbRequestSpec[],
      setResponseFns: React.Dispatch<
        React.SetStateAction<InferenceResponseAndResponseDict | null>
      >[]
    ) {
      batchedTdb({ subRequests })
        .then((responseData) => {
          if (responseData.inferenceSubResponses.length !== subRequests.length) {
            throw new Error(
              "Expected exactly " +
                subRequests.length +
                " inferenceSubResponses, but got " +
                responseData.inferenceSubResponses.length
            );
          }
          for (let i = 0; i < responseData.inferenceSubResponses.length; i++) {
            setResponseFns[i](responseData.inferenceSubResponses[i]);
          }
          setActivationServerErrorMessage(null);
        })
        .catch((error) => handleInferenceError(error, setActivationServerErrorMessage));
    }

    const newLeftRequest = buildTdbRequestSpec(leftPromptInferenceParams, commonInferenceParams);
    setLeftRequest(newLeftRequest);
    var newRightRequest = null;
    if (rightPromptInferenceParams !== null) {
      // We're comparing two prompts. The right request covers the right prompt and both prompts use
      // the same ablations (which are stored on the left).
      newRightRequest = buildTdbRequestSpec(
        {
          ...rightPromptInferenceParams,
          nodeAblations: leftPromptInferenceParams.nodeAblations,
          upstreamNodeToTrace: leftPromptInferenceParams.upstreamNodeToTrace,
        },
        commonInferenceParams
      );
    } else if (leftPromptInferenceParams.nodeAblations.length !== 0) {
      newRightRequest = buildTdbRequestSpec(
        {
          ...leftPromptInferenceParams,
          // The right request omits the ablations specified in the left request.
          nodeAblations: Array<NodeAblation>(),
        },
        commonInferenceParams
      );
    } else {
      // If there is no right prompt and no ablations, there is no need to make a separate right
      // request.
      newRightRequest = null;
    }
    setRightRequest(newRightRequest);

    const subRequests = [newLeftRequest];
    const setResponseFns = [setLeftResponse];
    if (newRightRequest !== null) {
      subRequests.push(newRightRequest);
      setResponseFns.push(setRightResponse);
    }
    performInference(subRequests, setResponseFns);
  }

  // TODO(sbills): Separate inference error handling for left vs. right, so a success for one
  // doesn't hide an error for the other.
  handleInferenceError(error: ApiError, setError: (errorMessage: string | null) => void) {
    var errorMessage = error.message;
    if (error.body && error.body.message) {
      errorMessage = error.body.message;
    }
    console.log(errorMessage);
    setError(errorMessage);
  }
}
