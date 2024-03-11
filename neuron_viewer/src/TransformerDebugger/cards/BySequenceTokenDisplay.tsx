import {
  MultipleTopKDerivedScalarsResponseData,
  InferenceAndTokenData,
  GroupId,
} from "../../client";
import React, { useMemo } from "react";
import JsonModal from "../common/JsonModal";
import { TokenAndScalar, DerivedScalarType } from "../../client";
import { ExplanatoryTooltip } from "../common/ExplanatoryTooltip";
import { ACT_TIMES_GRAD_EXPLANATION } from "../utils/explanations";
import TokenHeatmap from "../../tokenHeatmap";
import { POSITIVE_NEGATIVE_BOUNDARIES, POSITIVE_NEGATIVE_COLORS } from "../../colors";

function makeTokenAndScalarList(tokensAsStrings: string[], scalars: number[]): TokenAndScalar[] {
  const scalarsMax = Math.max(...scalars);
  const scalarsMin = Math.min(...scalars);
  const scale = Math.max(Math.abs(scalarsMax), Math.abs(scalarsMin));
  // console.log("scale", scale);
  const normalizedScalars = scalars.map((scalar) => scalar / (scale * 2) + 0.5);
  // console.log("normalizedScalars", normalizedScalars);
  let tokenAndScalarList: TokenAndScalar[] = [];
  for (let i = 0; i < tokensAsStrings.length; i++) {
    tokenAndScalarList.push({
      token: tokensAsStrings[i],
      scalar: scalars[i],
      normalizedScalar: normalizedScalars[i],
    });
  }
  return tokenAndScalarList;
}

function sumOverFirstDim(values: number[][]): number[] {
  let result = new Array(values.length).fill(0);
  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < values[i].length; j++) {
      result[i] += values[i][j];
    }
  }
  return result;
}

function sumOverSecondDim(values: number[][]): number[] {
  let result = new Array(values[0].length).fill(0);
  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < values[i].length; j++) {
      result[j] += values[i][j];
    }
  }
  return result;
}

type BySequenceTokenDisplayProps = {
  responseData: MultipleTopKDerivedScalarsResponseData;
  inferenceAndTokenData: InferenceAndTokenData;
};
export const BySequenceTokenDisplay: React.FC<BySequenceTokenDisplayProps> = ({
  responseData,
  inferenceAndTokenData,
}) => {
  const intermediateSumActivations = responseData.intermediateSumActivationsByDstByGroupId;
  const tokensAsStrings: string[] = inferenceAndTokenData.tokensAsStrings;

  const embActTimesGrad = useMemo(
    () =>
      makeTokenAndScalarList(
        tokensAsStrings,
        intermediateSumActivations[GroupId.ACT_TIMES_GRAD][DerivedScalarType.TOKEN_ATTRIBUTION]
          .value as unknown as number[]
      ),
    [tokensAsStrings, intermediateSumActivations]
  );

  const mlpActTimesGrad = useMemo(
    () =>
      makeTokenAndScalarList(
        tokensAsStrings,
        intermediateSumActivations[GroupId.ACT_TIMES_GRAD][DerivedScalarType.MLP_ACT_TIMES_GRAD]
          .value as unknown as number[]
      ),
    [tokensAsStrings, intermediateSumActivations]
  );

  const attendedFromToken = useMemo(
    () =>
      makeTokenAndScalarList(
        tokensAsStrings,
        sumOverFirstDim(
          intermediateSumActivations[GroupId.ACT_TIMES_GRAD][
            DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD
          ].value as unknown as number[][]
        )
      ),
    [tokensAsStrings, intermediateSumActivations]
  );

  const attendedToToken = useMemo(
    () =>
      makeTokenAndScalarList(
        tokensAsStrings,
        sumOverSecondDim(
          intermediateSumActivations[GroupId.ACT_TIMES_GRAD][
            DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD
          ].value as unknown as number[][]
        )
      ),
    [tokensAsStrings, intermediateSumActivations]
  );

  return (
    <div>
      <div className="flex justify-between">
        <h2 className="text-xl">Token display</h2>
        <JsonModal jsonData={{ responseData, inferenceAndTokenData }} />
      </div>
      <div>
        <ExplanatoryTooltip explanation={ACT_TIMES_GRAD_EXPLANATION}>
          <h3 className="text-lg flex">Estimated total effect, embeddings</h3>
        </ExplanatoryTooltip>
        <TokenHeatmap
          tokenSequence={embActTimesGrad}
          colors={POSITIVE_NEGATIVE_COLORS}
          boundaries={POSITIVE_NEGATIVE_BOUNDARIES}
          fixedWidth={true}
        />
      </div>
      <div>
        <ExplanatoryTooltip explanation={ACT_TIMES_GRAD_EXPLANATION}>
          <h3 className="text-lg flex">Estimated total effect, MLP layers</h3>
        </ExplanatoryTooltip>
        <TokenHeatmap
          tokenSequence={mlpActTimesGrad}
          colors={POSITIVE_NEGATIVE_COLORS}
          boundaries={POSITIVE_NEGATIVE_BOUNDARIES}
          fixedWidth={true}
        />
      </div>
      <div>
        <ExplanatoryTooltip explanation={ACT_TIMES_GRAD_EXPLANATION}>
          <h3 className="text-lg flex">
            Estimated total effect, attention layers, attended-from token
          </h3>
        </ExplanatoryTooltip>
        <TokenHeatmap
          tokenSequence={attendedFromToken}
          colors={POSITIVE_NEGATIVE_COLORS}
          boundaries={POSITIVE_NEGATIVE_BOUNDARIES}
          fixedWidth={true}
        />
      </div>
      <div>
        <ExplanatoryTooltip explanation={ACT_TIMES_GRAD_EXPLANATION}>
          <h3 className="text-lg flex">
            Estimated total effect, attention layers, attended-to token
          </h3>
        </ExplanatoryTooltip>
        <TokenHeatmap
          tokenSequence={attendedToToken}
          colors={POSITIVE_NEGATIVE_COLORS}
          boundaries={POSITIVE_NEGATIVE_BOUNDARIES}
          fixedWidth={true}
        />
      </div>
    </div>
  );
};
