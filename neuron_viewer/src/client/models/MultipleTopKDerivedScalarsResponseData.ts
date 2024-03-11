// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { MirroredNodeIndex } from "./MirroredNodeIndex";
import type { ProcessingResponseDataType } from "./ProcessingResponseDataType";
import type { Tensor0D } from "./Tensor0D";
import type { Tensor1D } from "./Tensor1D";
import type { Tensor2D } from "./Tensor2D";
import type { Tensor3D } from "./Tensor3D";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type MultipleTopKDerivedScalarsResponseData = {
  responseDataType?: ProcessingResponseDataType;
  activationsByGroupId: Record<string, Array<number>>;
  nodeIndices: Array<MirroredNodeIndex>;
  vocabTokenStringsForIndices?: Array<string>;
  intermediateSumActivationsByDstByGroupId: Record<
    string,
    Record<string, Tensor0D | Tensor1D | Tensor2D | Tensor3D>
  >;
};
