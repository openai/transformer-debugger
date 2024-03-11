// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { MirroredNodeIndex } from "./MirroredNodeIndex";

/**
 * A specification for tracing an upstream node.
 *
 * This data structure is used by the client. The server converts it to an AblationSpec.
 */
export type NodeAblation = {
  nodeIndex: MirroredNodeIndex;
  value: number;
};
