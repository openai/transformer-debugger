// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { BatchedRequest } from "../models/BatchedRequest";
import type { BatchedResponse } from "../models/BatchedResponse";
import type { BatchedTdbRequest } from "../models/BatchedTdbRequest";
import type { DerivedAttentionScalarsRequest } from "../models/DerivedAttentionScalarsRequest";
import type { DerivedAttentionScalarsResponse } from "../models/DerivedAttentionScalarsResponse";
import type { DerivedScalarsRequest } from "../models/DerivedScalarsRequest";
import type { DerivedScalarsResponse } from "../models/DerivedScalarsResponse";
import type { ModelInfoResponse } from "../models/ModelInfoResponse";
import type { MultipleTopKDerivedScalarsRequest } from "../models/MultipleTopKDerivedScalarsRequest";
import type { MultipleTopKDerivedScalarsResponse } from "../models/MultipleTopKDerivedScalarsResponse";

import type { CancelablePromise } from "../core/CancelablePromise";
import { OpenAPI } from "../core/OpenAPI";
import { request as __request } from "../core/request";

export class InferenceService {
  /**
   * Derived Scalars
   * @param requestBody
   * @returns DerivedScalarsResponse Successful Response
   * @throws ApiError
   */
  public static inferenceDerivedScalars(
    requestBody: DerivedScalarsRequest
  ): CancelablePromise<DerivedScalarsResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/derived_scalars",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Derived Attention Scalars
   * @param requestBody
   * @returns DerivedAttentionScalarsResponse Successful Response
   * @throws ApiError
   */
  public static inferenceDerivedAttentionScalars(
    requestBody: DerivedAttentionScalarsRequest
  ): CancelablePromise<DerivedAttentionScalarsResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/derived_attention_scalars",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Multiple Top K Derived Scalars
   * @param requestBody
   * @returns MultipleTopKDerivedScalarsResponse Successful Response
   * @throws ApiError
   */
  public static inferenceMultipleTopKDerivedScalars(
    requestBody: MultipleTopKDerivedScalarsRequest
  ): CancelablePromise<MultipleTopKDerivedScalarsResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/multiple_top_k_derived_scalars",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Batched
   * @param requestBody
   * @returns BatchedResponse Successful Response
   * @throws ApiError
   */
  public static inferenceBatched(requestBody: BatchedRequest): CancelablePromise<BatchedResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/batched",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Batched Tdb
   * @param requestBody
   * @returns BatchedResponse Successful Response
   * @throws ApiError
   */
  public static inferenceBatchedTdb(
    requestBody: BatchedTdbRequest
  ): CancelablePromise<BatchedResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/batched_tdb",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Model Info
   * @returns ModelInfoResponse Successful Response
   * @throws ApiError
   */
  public static inferenceModelInfo(): CancelablePromise<ModelInfoResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/model_info",
    });
  }
}
