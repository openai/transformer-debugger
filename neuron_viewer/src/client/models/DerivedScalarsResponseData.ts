// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { MirroredNodeIndex } from "./MirroredNodeIndex";
import type { ProcessingResponseDataType } from "./ProcessingResponseDataType";
import type { TopTokens } from "./TopTokens";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type DerivedScalarsResponseData = {
  responseDataType?: ProcessingResponseDataType;
  activations: Array<number>;
  normalizedActivations?: Array<number>;
  nodeIndices: Array<MirroredNodeIndex>;
  topTokens?: TopTokens;
};
