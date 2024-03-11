// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { DerivedScalarType } from "./DerivedScalarType";
import type { Dimension } from "./Dimension";
import type { PassType } from "./PassType";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type MultipleTopKDerivedScalarsRequestSpec = {
  specType?: MultipleTopKDerivedScalarsRequestSpec.specType;
  dstListByGroupId: Record<string, Array<DerivedScalarType>>;
  tokenIndex?: number;
  topAndBottomK?: number;
  passType?: PassType;
  dimensionsToKeepForIntermediateSum?: Array<Dimension>;
};

export namespace MultipleTopKDerivedScalarsRequestSpec {
  export enum specType {
    MULTIPLE_TOP_K_DERIVED_SCALARS_REQUEST_SPEC = "multiple_top_k_derived_scalars_request_spec",
  }
}
