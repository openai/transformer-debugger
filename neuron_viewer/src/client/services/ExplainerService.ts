// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ExplanationResult } from "../models/ExplanationResult";
import type { NodeIdAndDatasets } from "../models/NodeIdAndDatasets";
import type { ScoreRequest } from "../models/ScoreRequest";
import type { ScoreResult } from "../models/ScoreResult";

import type { CancelablePromise } from "../core/CancelablePromise";
import { OpenAPI } from "../core/OpenAPI";
import { request as __request } from "../core/request";

export class ExplainerService {
  /**
   * Explain
   * @param requestBody
   * @returns ExplanationResult Successful Response
   * @throws ApiError
   */
  public static explainerExplain(
    requestBody: NodeIdAndDatasets
  ): CancelablePromise<ExplanationResult> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/explain",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Score
   * @param requestBody
   * @returns ScoreResult Successful Response
   * @throws ApiError
   */
  public static explainerScore(requestBody: ScoreRequest): CancelablePromise<ScoreResult> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/score",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }
}
