import {
  InferenceService,
  DerivedScalarsRequest,
  BatchedRequest,
  ModelInfoResponse,
  BatchedTdbRequest,
  DerivedAttentionScalarsRequest,
} from "../client";

export const derivedScalars = async (request: DerivedScalarsRequest) => {
  return await InferenceService.inferenceDerivedScalars(request);
};

export const derivedAttentionScalars = async (request: DerivedAttentionScalarsRequest) => {
  return await InferenceService.inferenceDerivedAttentionScalars(request);
};

export const combinedInference = async (request: BatchedRequest) => {
  return await InferenceService.inferenceBatched(request);
};

export const batchedTdb = async (request: BatchedTdbRequest) => {
  return await InferenceService.inferenceBatchedTdb(request);
};

export const getModelInfo: () => Promise<ModelInfoResponse> = async () => {
  return await InferenceService.inferenceModelInfo();
};
