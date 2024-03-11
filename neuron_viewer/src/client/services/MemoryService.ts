// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CancelablePromise } from "../core/CancelablePromise";
import { OpenAPI } from "../core/OpenAPI";
import { request as __request } from "../core/request";

export class MemoryService {
  /**
   * Dump Memory Snapshot
   * @returns string Successful Response
   * @throws ApiError
   */
  public static memoryDumpMemorySnapshot(): CancelablePromise<string> {
    return __request(OpenAPI, {
      method: "GET",
      url: "/dump_memory_snapshot",
    });
  }
}
