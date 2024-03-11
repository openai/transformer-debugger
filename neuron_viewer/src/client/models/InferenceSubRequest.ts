// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { DerivedAttentionScalarsRequestSpec } from "./DerivedAttentionScalarsRequestSpec";
import type { DerivedScalarsRequestSpec } from "./DerivedScalarsRequestSpec";
import type { InferenceRequestSpec } from "./InferenceRequestSpec";
import type { MultipleTopKDerivedScalarsRequestSpec } from "./MultipleTopKDerivedScalarsRequestSpec";
import type { ScoredTokensRequestSpec } from "./ScoredTokensRequestSpec";
import type { TokenPairAttributionRequestSpec } from "./TokenPairAttributionRequestSpec";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type InferenceSubRequest = {
  inferenceRequestSpec: InferenceRequestSpec;
  processingRequestSpecByName?: Record<
    string,
    | MultipleTopKDerivedScalarsRequestSpec
    | DerivedScalarsRequestSpec
    | DerivedAttentionScalarsRequestSpec
    | ScoredTokensRequestSpec
    | TokenPairAttributionRequestSpec
  >;
};
