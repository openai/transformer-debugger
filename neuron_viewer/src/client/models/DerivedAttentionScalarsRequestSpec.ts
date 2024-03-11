// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { DerivedScalarType } from "./DerivedScalarType";
import type { NodeIdAndDatasets } from "./NodeIdAndDatasets";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type DerivedAttentionScalarsRequestSpec = {
  specType?: DerivedAttentionScalarsRequestSpec.specType;
  dst: DerivedScalarType;
  layerIndex?: number;
  activationIndex: number;
  normalizeActivationsUsingNeuronRecord?: NodeIdAndDatasets;
};

export namespace DerivedAttentionScalarsRequestSpec {
  export enum specType {
    DERIVED_ATTENTION_SCALARS_REQUEST_SPEC = "derived_attention_scalars_request_spec",
  }
}
