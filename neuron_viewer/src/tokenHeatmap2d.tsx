import { useState } from "react";
import { TokenSequenceAndAttentionScalars } from "./types";
import { formatToken } from "./tokenRendering";
import {
  getInterpolatedColor,
  BLANK_COLOR,
  MAX_IN_COLOR,
  DEFAULT_BOUNDARIES,
  MAX_OUT_COLOR,
  subtractiveMix,
} from "./colors";

/**
 * What's visualized here is the norm of the vector written between any two pairs
 * of tokens by an attention head. The token sequences shown contain the pairs of
 * tokens for which this vector norm was largest.
 *
 * Norm of the write vector going OUT of a token (the attended-TO token) is shown
 * in cyan. Norm of the write vector going IN to a token (the attended-FROM token)
 * is shown in magenta. Mouseover allows you to look at inputs to the current token,
 * while no mouseover summarizes each token by its max inputs and outputs.
 *
 * Total inputs and outputs (i.e. sum rather than max) are computed but not currently
 * visualized.
 */

type TokenHeatmapProps = {
  tokenSequenceAndAttentionScalars?: TokenSequenceAndAttentionScalars; // undefined means we're rendering an empty box while loading.
  onClick?: (index: number) => void;
};

const TokenHeatmap2d: React.FC<TokenHeatmapProps> = ({
  tokenSequenceAndAttentionScalars,
  onClick,
}) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const handleMouseOver = (index: number) => {
    setHoverIndex(index);
  };

  const handleMouseOut = () => {
    setHoverIndex(null);
  };

  // if true, then color by the norm of the vector written to each token
  // if false, then color by the norm of the vector written from each token
  const showScalarsInOnMouseOver = false;

  return (
    <div className="block" style={{ width: "100%" }}>
      {tokenSequenceAndAttentionScalars &&
        tokenSequenceAndAttentionScalars.map(
          (
            {
              token,
              scalars,
              maxScalarIn,
              normalizedMaxScalarIn,
              maxScalarOut,
              normalizedMaxScalarOut,
            },
            i
          ) => {
            let colorIn; // white to cyan
            let colorOut; // white to magenta

            // without mouseover, colorIn is the max of values going into the current token
            // with mouseover of the current token, colorIn is the max of values going into the current token
            // with mouseover of a different token, colorIn is white
            if (hoverIndex === null) {
              colorIn = getInterpolatedColor(
                [BLANK_COLOR, MAX_IN_COLOR],
                DEFAULT_BOUNDARIES,
                (normalizedMaxScalarIn || maxScalarIn)!
              );
            } else {
              if (showScalarsInOnMouseOver) {
                if (i === hoverIndex) {
                  colorIn = getInterpolatedColor(
                    [BLANK_COLOR, MAX_IN_COLOR],
                    DEFAULT_BOUNDARIES,
                    (normalizedMaxScalarIn || maxScalarIn)!
                  );
                } else {
                  colorIn = BLANK_COLOR;
                }
              } else {
                // showing scalars out from the moused over token
                if (i >= hoverIndex) {
                  colorIn = getInterpolatedColor(
                    [BLANK_COLOR, MAX_IN_COLOR],
                    DEFAULT_BOUNDARIES,
                    tokenSequenceAndAttentionScalars[i]?.normalizedScalars?.[hoverIndex] ||
                      tokenSequenceAndAttentionScalars[i]?.scalars?.[hoverIndex] ||
                      0
                  );
                } else {
                  colorIn = BLANK_COLOR;
                }
              }
            }

            // without mouseover, colorOut is the max of values going out of the current token
            // with mouseover of the current token or later tokens, colorOut is the value going from the current token to the mouseover token
            // with mouseover of an earlier token, colorOut is white
            if (hoverIndex === null) {
              colorOut = getInterpolatedColor(
                [BLANK_COLOR, MAX_OUT_COLOR],
                DEFAULT_BOUNDARIES,
                (normalizedMaxScalarOut || maxScalarOut)!
              );
            } else {
              if (showScalarsInOnMouseOver) {
                if (i <= hoverIndex) {
                  colorOut = getInterpolatedColor(
                    [BLANK_COLOR, MAX_OUT_COLOR],
                    DEFAULT_BOUNDARIES,
                    tokenSequenceAndAttentionScalars[hoverIndex]?.normalizedScalars?.[i] ||
                      tokenSequenceAndAttentionScalars[hoverIndex]?.scalars?.[i] ||
                      0
                  );
                } else {
                  colorOut = BLANK_COLOR;
                }
              } else {
                // showing scalars out from the moused over token
                if (i === hoverIndex) {
                  colorOut = getInterpolatedColor(
                    [BLANK_COLOR, MAX_OUT_COLOR],
                    DEFAULT_BOUNDARIES,
                    (normalizedMaxScalarOut || maxScalarOut)!
                  );
                } else {
                  colorOut = BLANK_COLOR;
                }
              }
            }

            const colorSummary = subtractiveMix(colorIn, colorOut);

            const outputsOfIndexI = tokenSequenceAndAttentionScalars
              .slice(i)
              .map((item) => item?.scalars?.[i]);

            let tooltipScalars;
            if (showScalarsInOnMouseOver) {
              tooltipScalars = scalars; // inputs to index i
            } else {
              tooltipScalars = outputsOfIndexI;
            }

            return (
              <span
                key={i}
                // We deliberately don't use React tooltips here. They appear immediately on
                // mouseover, making it hard to see the colors changing to reflect the attention
                // from the hovered-over token. The regular title-based tooltips are better, since
                // they only appear after a delay, and since their formatting is more compact.
                title={`Scalar: ${tooltipScalars.map((item) => item.toFixed(2)).join(", ")}`}
                className="whitespace-pre-wrap"
                style={{
                  transition: "500ms ease-in all",
                  background: `rgba(${colorSummary.r}, ${colorSummary.g}, ${colorSummary.b}, 0.5)`,
                }}
                onClick={() => onClick && onClick(i)}
                onMouseOver={() => handleMouseOver(i)}
                onMouseOut={handleMouseOut}
              >
                {formatToken(token, /* dotsForSpaces= */ false)}
              </span>
            );
          }
        )}
    </div>
  );
};

export default TokenHeatmap2d;
