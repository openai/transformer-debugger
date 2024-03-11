// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { AttentionTraceType } from "./AttentionTraceType";
import type { MirroredNodeIndex } from "./MirroredNodeIndex";
import type { MirroredTraceConfig } from "./MirroredTraceConfig";

/**
 * A specification for tracing a node.
 *
 * This data structure is used by the client. The server converts it to an activation index and
 * an ablation spec.
 *
 * In the case of tracing through attention value, there can be up to two NodeToTrace
 * objects: one upstream and one downstream. First, a gradient is computed with respect to the
 * downstream node. Then, the direct effect of the upstream (attention) node on that downstream
 * node is computed. Then, the gradient is computed with respect to that direct effect, propagated
 * through V
 */
export type NodeToTrace = {
  nodeIndex: MirroredNodeIndex;
  attentionTraceType?: AttentionTraceType;
  downstreamTraceConfig?: MirroredTraceConfig;
};
