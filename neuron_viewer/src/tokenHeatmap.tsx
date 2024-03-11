import { Tooltip } from "@nextui-org/react";
import { formatToken } from "./tokenRendering";
import { TokenSequenceAndScalars } from "./types";
import { Color, DEFAULT_BOUNDARIES, DEFAULT_COLORS, getInterpolatedColor } from "./colors";

type TokenHeatmapProps = {
  tokenSequence?: TokenSequenceAndScalars; // undefined means we're rendering an empty box while loading.
  onClick?: (index: number) => void;
  colors?: Color[];
  boundaries?: number[];
  fixedWidth?: boolean;
};

const TokenHeatmap: React.FC<TokenHeatmapProps> = ({
  tokenSequence,
  onClick,
  colors,
  boundaries,
  fixedWidth,
}) => {
  return (
    <div className="block" style={{ width: "100%" }}>
      {tokenSequence &&
        tokenSequence.map(({ token, scalar, normalizedScalar }, i) => {
          const color = getInterpolatedColor(
            colors || DEFAULT_COLORS,
            boundaries || DEFAULT_BOUNDARIES,
            normalizedScalar || scalar
          );
          return (
            <Tooltip content={`Activation: ${scalar.toFixed(2)} Index: ${i}`}>
              <span
                key={i}
                className={`whitespace-pre-wrap` + fixedWidth ? " font-mono" : ""}
                style={{
                  transition: "500ms ease-in all",
                  background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`,
                }}
                onClick={() => onClick && onClick(i)}
              >
                {formatToken(token, /* dotsForSpaces= */ false)}
              </span>
            </Tooltip>
          );
        })}
    </div>
  );
};
export default TokenHeatmap;
