// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { NodeType } from "./NodeType";
import type { PassType } from "./PassType";

/**
 * This class mirrors the fields of NodeIndex without default values.
 */
export type MirroredNodeIndex = {
  nodeType: NodeType;
  tensorIndices: Array<number>;
  layerIndex?: number;
  passType: PassType;
};
