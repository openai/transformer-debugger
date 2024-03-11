// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { AblationSpec } from "./AblationSpec";
import type { LossFnConfig } from "./LossFnConfig";
import type { MirroredActivationIndex } from "./MirroredActivationIndex";
import type { MirroredTraceConfig } from "./MirroredTraceConfig";

/**
 * The minimum specification for performing a forward and/or backward pass on a model, with hooks at some set of layers.
 */
export type InferenceRequestSpec = {
  prompt: string;
  ablationSpecs?: Array<AblationSpec>;
  lossFnConfig?: LossFnConfig;
  traceConfig?: MirroredTraceConfig;
  activationIndexForWithinLayerGrad?: MirroredActivationIndex;
};
