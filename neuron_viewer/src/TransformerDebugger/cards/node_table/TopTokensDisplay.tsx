import { TopTokens, TokenAndScalar } from "../../../client";
import { Tooltip } from "@nextui-org/react";
import { ExplanatoryTooltip } from "../../common/ExplanatoryTooltip";
import { renderTokenOnGray } from "../../../tokenRendering";

const renderTokenList = (
  title: string,
  explanation: string,
  tokens: TokenAndScalar[],
  maxTokens?: number
) => (
  <span className="text-sm text-gray-700">
    <span>
      <ExplanatoryTooltip explanation={explanation}>
        <span>
          <strong>{title}:</strong>
        </span>
      </ExplanatoryTooltip>
    </span>
    {tokens.slice(0, maxTokens).map((token, idx) => {
      return (
        <Tooltip key={idx} content={token.scalar.toFixed(2)}>
          {renderTokenOnGray(token.token, idx)}
        </Tooltip>
      );
    })}
  </span>
);

function whichSidesToDisplay(
  leftSideData: TopTokens | null,
  rightSideData: TopTokens | null,
  maxTokens?: number
): { displayLeftSide: boolean; displayRightSide: boolean } {
  // Display both sides unless all the tokens are the same. If they are the same, then show the side
  // with larger magnitude on the first token (to avoid showing a side with all 0s)
  let displayLeftSide = leftSideData !== null;
  let displayRightSide = rightSideData !== null;
  if (leftSideData && rightSideData) {
    const leftTopToken = leftSideData.top[0];
    const rightTopToken = rightSideData.top[0];
    if (Math.abs(leftTopToken.scalar) <= 0.01) {
      return { displayLeftSide: false, displayRightSide: true };
    }
    if (Math.abs(rightTopToken.scalar) <= 0.01) {
      return { displayLeftSide: true, displayRightSide: false };
    }
    const leftTopTokens = leftSideData.top.slice(0, maxTokens).map((token) => token.token);
    const rightTopTokens = rightSideData.top.slice(0, maxTokens).map((token) => token.token);
    const leftBottomTokens = leftSideData.bottom.slice(0, maxTokens).map((token) => token.token);
    const rightBottomTokens = rightSideData.bottom.slice(0, maxTokens).map((token) => token.token);
    const topTokensAreEqual =
      leftTopTokens.length === rightTopTokens.length &&
      leftTopTokens.every((token, index) => token === rightTopTokens[index]);
    const bottomTokensAreEqual =
      leftBottomTokens.length === rightBottomTokens.length &&
      leftBottomTokens.every((token, index) => token === rightBottomTokens[index]);
    if (topTokensAreEqual && bottomTokensAreEqual) {
      displayLeftSide = Math.abs(leftTopToken.scalar) > Math.abs(rightTopToken.scalar);
      displayRightSide = !displayLeftSide;
    }
    return { displayLeftSide, displayRightSide };
  }
  return { displayLeftSide, displayRightSide };
}

export const TopTokensDisplay: React.FC<{
  leftSideData: TopTokens | null;
  rightSideData: TopTokens | null;
  label: string;
  explanations: { increase: string; decrease: string };
}> = ({ leftSideData, rightSideData, label, explanations }) => {
  const { displayLeftSide, displayRightSide } = whichSidesToDisplay(leftSideData, rightSideData);
  const leftTopTokens = leftSideData?.top;
  const leftBottomTokens = leftSideData?.bottom;
  const rightTopTokens = rightSideData?.top;
  const rightBottomTokens = rightSideData?.bottom;
  const leftTitlePrefix = "Left ";
  const rightTitlePrefix = "Right ";
  return (
    <div>
      {displayLeftSide && (
        <div>
          {leftTopTokens &&
            renderTokenList(
              `${leftTitlePrefix}${label} top`,
              explanations.increase,
              leftTopTokens,
              10
            )}
          {leftBottomTokens &&
            renderTokenList("bottom", explanations.decrease, leftBottomTokens, 10)}
        </div>
      )}
      {displayRightSide && (
        <div>
          {rightTopTokens &&
            renderTokenList(
              `${rightTitlePrefix}${label} top`,
              explanations.increase,
              rightTopTokens,
              10
            )}
          {rightBottomTokens &&
            renderTokenList("bottom", explanations.decrease, rightBottomTokens, 10)}
        </div>
      )}
    </div>
  );
};
