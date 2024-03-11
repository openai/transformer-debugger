// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { TokenAndScalar } from "./TokenAndScalar";

/**
 * Contains two lists of tokens and associated scalars: one for the highest-scoring tokens and one
 * for the lowest-scoring tokens, according to some way of scoring tokens. For example, this could
 * be used to represent the top upvoted and downvoted "logit lens" tokens. An instance of this
 * class is scoped to a single node. The set of tokens eligible for scoring is typically just the
 * model's entire vocabulary. Each list is sorted from largest to smallest absolute value for the
 * associated scalar.
 */
export type TopTokens = {
  top: Array<TokenAndScalar>;
  bottom: Array<TokenAndScalar>;
};
