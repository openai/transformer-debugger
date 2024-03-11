import type { InferenceAndTokenData, InferenceResponseAndResponseDict } from "../../client";

export function getSubResponse<T>(
  responseData: InferenceResponseAndResponseDict | null,
  requestSpecName: string
): T | null {
  if (!responseData) {
    return null;
  }
  return responseData.processingResponseDataByName![requestSpecName] as T;
}

export function getInferenceAndTokenData(
  responseData: InferenceResponseAndResponseDict | null
): InferenceAndTokenData | null {
  if (!responseData) {
    return null;
  }
  return responseData.inferenceResponse.inferenceAndTokenData;
}
