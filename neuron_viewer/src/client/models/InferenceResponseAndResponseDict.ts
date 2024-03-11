// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { DerivedAttentionScalarsResponseData } from "./DerivedAttentionScalarsResponseData";
import type { DerivedScalarsResponseData } from "./DerivedScalarsResponseData";
import type { InferenceResponse } from "./InferenceResponse";
import type { MultipleTopKDerivedScalarsResponseData } from "./MultipleTopKDerivedScalarsResponseData";
import type { ScoredTokensResponseData } from "./ScoredTokensResponseData";
import type { TokenPairAttributionResponseData } from "./TokenPairAttributionResponseData";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type InferenceResponseAndResponseDict = {
  inferenceResponse: InferenceResponse;
  processingResponseDataByName?: Record<
    string,
    | MultipleTopKDerivedScalarsResponseData
    | DerivedScalarsResponseData
    | DerivedAttentionScalarsResponseData
    | ScoredTokensResponseData
    | TokenPairAttributionResponseData
  >;
};
