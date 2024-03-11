// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AttentionHeadRecordResponse } from "../models/AttentionHeadRecordResponse";
import type { AttributedScoredExplanation } from "../models/AttributedScoredExplanation";
import type { ExistingExplanationsRequest } from "../models/ExistingExplanationsRequest";
import type { NeuronDatasetMetadata } from "../models/NeuronDatasetMetadata";
import type { NeuronRecordResponse } from "../models/NeuronRecordResponse";
import type { NodeIdAndDatasets } from "../models/NodeIdAndDatasets";

import type { CancelablePromise } from "../core/CancelablePromise";
import { OpenAPI } from "../core/OpenAPI";
import { request as __request } from "../core/request";

export class ReadService {
  /**
   * Existing Explanations
   * @param requestBody
   * @returns AttributedScoredExplanation Successful Response
   * @throws ApiError
   */
  public static readExistingExplanations(
    requestBody: ExistingExplanationsRequest
  ): CancelablePromise<Array<AttributedScoredExplanation>> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/existing_explanations",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Neuron Record
   * @param requestBody
   * @returns NeuronRecordResponse Successful Response
   * @throws ApiError
   */
  public static readNeuronRecord(
    requestBody: NodeIdAndDatasets
  ): CancelablePromise<NeuronRecordResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/neuron_record",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Attention Head Record
   * @param requestBody
   * @returns AttentionHeadRecordResponse Successful Response
   * @throws ApiError
   */
  public static readAttentionHeadRecord(
    requestBody: NodeIdAndDatasets
  ): CancelablePromise<AttentionHeadRecordResponse> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/attention_head_record",
      body: requestBody,
      mediaType: "application/json",
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Neuron Datasets Metadata
   * @returns NeuronDatasetMetadata Successful Response
   * @throws ApiError
   */
  public static readNeuronDatasetsMetadata(): CancelablePromise<Array<NeuronDatasetMetadata>> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/neuron_datasets_metadata",
    });
  }
}
