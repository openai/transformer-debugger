import React, { useMemo } from "react";

import {
  MultipleTopKDerivedScalarsResponseData,
  InferenceAndTokenData,
  NodeType,
  GroupId,
} from "../../client";
import { joinIndices } from "../utils/nodes";
import JsonModal from "../common/JsonModal";
import { AgGridReact } from "ag-grid-react";
import {
  compareWithUndefinedLast,
  diffOptionalNumbers,
  formatFloat,
  formatFloatWithZeroPoint,
} from "../utils/numbers";
import { ColDef, ColGroupDef } from "ag-grid-community";
import "ag-grid-community/styles/ag-grid.css"; // Core grid CSS, always needed
import "ag-grid-community/styles/ag-theme-alpine.css"; // Optional theme CSS
import { renderToken } from "../../tokenRendering";
import { PromptInferenceParams } from "./inference_params/inferenceParams";

type LogitsDisplayProps = {
  leftPromptInferenceParams: PromptInferenceParams;
  rightPromptInferenceParams: PromptInferenceParams | null;
  leftResponseData: MultipleTopKDerivedScalarsResponseData;
  rightResponseData: MultipleTopKDerivedScalarsResponseData | null;
  leftInferenceAndTokenData: InferenceAndTokenData;
  rightInferenceAndTokenData: InferenceAndTokenData | null;
};

type LogitDatum = {
  token: string;
  label?: string;
  rightLogit?: number;
  leftLogit?: number;
  diffLogit?: number;
};

export const LogitsDisplay: React.FC<LogitsDisplayProps> = ({
  rightResponseData,
  leftResponseData,
  leftInferenceAndTokenData,
  rightInferenceAndTokenData,
  leftPromptInferenceParams,
  rightPromptInferenceParams,
}) => {
  const collatedLogitData: LogitDatum[] = useCollatedLogitData(rightResponseData, leftResponseData);
  const rightPromptInferenceParamsDefined = !!rightPromptInferenceParams;
  // For displaying logits, we subtract the first target token's logit on each side from the logits
  // so that the first target token's logit is 0. This makes the table easier to read.
  const [tokensOfInterestData, otherTokensData, leftBaselineLogit, rightBaselineLogit] =
    useMemo(() => {
      const tokensOfInterest = [
        ...leftPromptInferenceParams.targetTokens,
        ...leftPromptInferenceParams.distractorTokens,
      ];
      if (rightPromptInferenceParamsDefined) {
        tokensOfInterest.push(
          ...rightPromptInferenceParams.targetTokens,
          ...rightPromptInferenceParams.distractorTokens
        );
      }
      const tokensOfInterestData = collatedLogitData.filter((datum) =>
        tokensOfInterest.includes(datum.token)
      );
      // Sort order: left target, left distractor, right target, right distractor
      tokensOfInterestData.sort((a, b) => {
        const aIsLeftTarget = leftPromptInferenceParams.targetTokens.includes(a.token);
        const aIsLeftDistractor = leftPromptInferenceParams.distractorTokens.includes(a.token);
        const bIsLeftTarget = leftPromptInferenceParams.targetTokens.includes(b.token);
        const bIsLeftDistractor = leftPromptInferenceParams.distractorTokens.includes(b.token);
        const bIsRightTarget =
          rightPromptInferenceParamsDefined &&
          rightPromptInferenceParams.targetTokens.includes(b.token);
        const bIsRightDistractor =
          rightPromptInferenceParamsDefined &&
          rightPromptInferenceParams.distractorTokens.includes(b.token);

        if (aIsLeftTarget && !bIsLeftTarget) return -1;
        if (aIsLeftDistractor && !bIsLeftTarget && !bIsLeftDistractor) return -1;
        if (bIsRightTarget && !aIsLeftTarget && !aIsLeftDistractor) return 1;
        if (bIsRightDistractor && !aIsLeftTarget && !aIsLeftDistractor) return 1;
        return 0;
      });
      const otherTokensData = collatedLogitData.filter(
        (datum) => !tokensOfInterest.includes(datum.token)
      );
      const leftBaselineLogit = tokensOfInterestData[0]?.leftLogit ?? 0;
      const rightBaselineLogit = tokensOfInterestData[0]?.rightLogit ?? 0;
      tokensOfInterestData.forEach((datum) => {
        const isLeftTarget = leftPromptInferenceParams.targetTokens.includes(datum.token);
        const isRightTarget =
          rightPromptInferenceParamsDefined &&
          rightPromptInferenceParams.targetTokens.includes(datum.token);
        const isLeftDistractor = leftPromptInferenceParams.distractorTokens.includes(datum.token);
        const isRightDistractor =
          rightPromptInferenceParamsDefined &&
          rightPromptInferenceParams.distractorTokens.includes(datum.token);

        if (isLeftTarget && (!rightPromptInferenceParamsDefined || isRightTarget)) {
          datum.label = "target";
        } else if (isLeftTarget) {
          datum.label = "left target";
        } else if (isRightTarget) {
          datum.label = "right target";
        }

        if (isLeftDistractor && (!rightPromptInferenceParamsDefined || isRightDistractor)) {
          datum.label = "distractor";
        } else if (isLeftDistractor) {
          datum.label = "left distractor";
        } else if (isRightDistractor) {
          datum.label = "right distractor";
        }
      });
      return [tokensOfInterestData, otherTokensData, leftBaselineLogit, rightBaselineLogit];
    }, [
      collatedLogitData,
      leftPromptInferenceParams.targetTokens,
      leftPromptInferenceParams.distractorTokens,
      rightPromptInferenceParams?.targetTokens,
      rightPromptInferenceParams?.distractorTokens,
      rightPromptInferenceParamsDefined,
    ]);

  const lossDescription = useLossDescription(
    leftPromptInferenceParams,
    leftInferenceAndTokenData,
    rightInferenceAndTokenData
  );

  const columnDefs: (ColDef<LogitDatum, any> | ColGroupDef<LogitDatum>)[] = useMemo(() => {
    const defaultFloatColDefs: ColDef<LogitDatum, any> = {
      valueFormatter: (params: any) => formatFloat(params.value),
      resizable: true,
      width: 150,
      sortingOrder: ["desc", "asc"],
      sortable: true,
      comparator: compareWithUndefinedLast,
    };

    const rightColumns: (ColDef<LogitDatum, any> | ColGroupDef<LogitDatum>)[] = rightResponseData
      ? [
          {
            ...defaultFloatColDefs,
            field: "diffLogit",
            headerName: "Diff (left - right logit)",
            valueFormatter: (params: any) =>
              formatFloatWithZeroPoint(
                params.value,
                leftBaselineLogit - rightBaselineLogit,
                /* numDecimalPlaces= */ 3
              ),
          },
          {
            ...defaultFloatColDefs,
            field: "rightLogit",
            headerName: "Right logit",
            valueFormatter: (params: any) =>
              formatFloatWithZeroPoint(params.value, rightBaselineLogit),
          },
        ]
      : [];
    const leftColumns: (ColDef<LogitDatum, any> | ColGroupDef<LogitDatum>)[] = [
      {
        field: "token",
        headerName: "Token",
        width: 200,
        cellRenderer: (params: any) => {
          return (
            <div>
              {renderToken(params.value)}
              {params.data.label && (
                <span className="ml-2 text-xs text-gray-500 font-sans">({params.data.label})</span>
              )}
            </div>
          );
        },
      },
      {
        ...defaultFloatColDefs,
        field: "leftLogit",
        headerName: rightResponseData ? "Left logit" : "Logit",
        initialSort: "desc",
        valueFormatter: (params: any) => formatFloatWithZeroPoint(params.value, leftBaselineLogit),
      },
    ];
    return [...leftColumns, ...rightColumns];
  }, [rightResponseData, leftBaselineLogit, rightBaselineLogit]);

  return (
    <div>
      <h2 className="text-xl">Next token candidates</h2>
      <div className="flex justify-between">
        {lossDescription}
        <JsonModal
          jsonData={{
            rightResponseData,
            leftResponseData,
            leftInferenceAndTokenData,
            rightInferenceAndTokenData,
            inferenceParams: leftPromptInferenceParams,
          }}
        />
      </div>
      <div className="ag-theme-alpine mt-2 mb-2" style={{ width: "800px", height: "400px" }}>
        <AgGridReact
          columnDefs={columnDefs}
          rowData={otherTokensData}
          pinnedTopRowData={tokensOfInterestData}
        />
      </div>
    </div>
  );
};

function useCollatedLogitData(
  rightResponseData: MultipleTopKDerivedScalarsResponseData | null,
  leftResponseData: MultipleTopKDerivedScalarsResponseData
): LogitDatum[] {
  return useMemo(() => {
    const lookupTable = joinIndices(
      rightResponseData?.nodeIndices ?? [],
      leftResponseData.nodeIndices
    );
    const collatedLogitData: LogitDatum[] = lookupTable.nodeIndices.map((nodeIndex, i) => {
      const rightIndex = lookupTable.rightArrayIndices[i];
      const leftIndex = lookupTable.leftArrayIndices[i];
      const rightLogit =
        rightIndex === undefined || !rightResponseData
          ? undefined
          : rightResponseData.activationsByGroupId[GroupId.LOGITS][rightIndex];
      const leftLogit =
        leftIndex === undefined
          ? undefined
          : leftResponseData.activationsByGroupId[GroupId.LOGITS][leftIndex];
      const diffLogit = diffOptionalNumbers(leftLogit, rightLogit);
      let token =
        leftIndex === undefined || !leftResponseData
          ? undefined
          : leftResponseData.vocabTokenStringsForIndices![leftIndex];
      if (token === undefined && rightIndex !== undefined && rightResponseData) {
        token = rightResponseData?.vocabTokenStringsForIndices![rightIndex];
      }
      return {
        token: token ?? "<missing>",
        rightLogit,
        leftLogit,
        diffLogit,
      };
    });
    return collatedLogitData;
  }, [rightResponseData, leftResponseData]);
}

function useLossDescription(
  inferenceParams: PromptInferenceParams,
  leftInferenceAndTokenData: InferenceAndTokenData,
  rightInferenceAndTokenData: InferenceAndTokenData | null
) {
  return useMemo(() => {
    let name = "";
    let leftValue = undefined;
    let rightValue = undefined;
    let diffValue = undefined;
    let description = "";
    if (inferenceParams.upstreamNodeToTrace !== null) {
      const nodeType = inferenceParams.upstreamNodeToTrace.nodeIndex.nodeType;
      description = nodeType === NodeType.ATTENTION_HEAD ? "(QK Inner Product)" : "(Activation)";
      name = "activation";
      leftValue = leftInferenceAndTokenData.activationValueForBackwardPass;
      rightValue = rightInferenceAndTokenData?.activationValueForBackwardPass;
    } else {
      name = "loss";
      leftValue = leftInferenceAndTokenData.loss;
      rightValue = rightInferenceAndTokenData?.loss;
      description = "(Target logit - distractor token logit, after layernorm)";
    }
    if (leftValue !== undefined && rightValue !== undefined) {
      diffValue = leftValue - rightValue;
    }
    diffValue = diffOptionalNumbers(leftValue, rightValue);
    return (
      <div>
        <div>
          <span className="mr-2">
            {rightInferenceAndTokenData ? "Left" : ""} {name}: {formatFloat(leftValue)}
          </span>
          {rightInferenceAndTokenData && (
            <>
              <span>
                Right {name}: {formatFloat(rightValue)}
              </span>
              <span className="ml-2">
                Diff {name}: {formatFloat(diffValue)}
              </span>
            </>
          )}
        </div>
        <div>{description}</div>
      </div>
    );
  }, [
    inferenceParams.upstreamNodeToTrace,
    leftInferenceAndTokenData.loss,
    leftInferenceAndTokenData.activationValueForBackwardPass,
    rightInferenceAndTokenData,
  ]);
}
