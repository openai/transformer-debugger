// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { TokenScoringType } from "./TokenScoringType";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type ScoredTokensRequestSpec = {
  specType?: ScoredTokensRequestSpec.specType;
  tokenScoringType: TokenScoringType;
  numTokens: number;
  dependsOnSpecName: string;
};

export namespace ScoredTokensRequestSpec {
  export enum specType {
    SCORED_TOKENS_REQUEST_SPEC = "scored_tokens_request_spec",
  }
}
