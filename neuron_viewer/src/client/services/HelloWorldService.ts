// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it.

/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { GroupId } from "../models/GroupId";

import type { CancelablePromise } from "../core/CancelablePromise";
import { OpenAPI } from "../core/OpenAPI";
import { request as __request } from "../core/request";

export class HelloWorldService {
  /**
   * Read Root
   * @returns string Successful Response
   * @throws ApiError
   */
  public static helloWorldReadRoot(): CancelablePromise<Record<string, string>> {
    return __request(OpenAPI, {
      method: "GET",
      url: "/",
    });
  }

  /**
   * Force Client Code Generation
   * @param groupId
   * @returns any Successful Response
   * @throws ApiError
   */
  public static helloWorldForceClientCodeGeneration(groupId: GroupId): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: "GET",
      url: "/force_client_code_generation",
      query: {
        group_id: groupId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
}
