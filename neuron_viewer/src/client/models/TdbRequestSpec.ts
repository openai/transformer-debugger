// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ComponentTypeForAttention } from "./ComponentTypeForAttention";
import type { ComponentTypeForMlp } from "./ComponentTypeForMlp";
import type { NodeAblation } from "./NodeAblation";
import type { NodeToTrace } from "./NodeToTrace";

/**
 * Base model that will automatically generate camelCase aliases for fields. Python code can use
 * either snake_case or camelCase names. When Typescript code is generated, it will only use the
 * camelCase names.
 */
export type TdbRequestSpec = {
  specType?: TdbRequestSpec.specType;
  prompt: string;
  targetTokens: Array<string>;
  distractorTokens: Array<string>;
  componentTypeForMlp: ComponentTypeForMlp;
  componentTypeForAttention: ComponentTypeForAttention;
  topAndBottomKForNodeTable: number;
  hideEarlyLayersWhenAblating: boolean;
  nodeAblations?: Array<NodeAblation>;
  upstreamNodeToTrace?: NodeToTrace;
  downstreamNodeToTrace?: NodeToTrace;
};

export namespace TdbRequestSpec {
  export enum specType {
    TDB_REQUEST_SPEC = "tdb_request_spec",
  }
}
