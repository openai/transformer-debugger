import { TokenSequenceAndScalars } from "./types";
import TokenHeatmap from "./tokenHeatmap";

export type HeatmapGridProps = {
  tokenSequences: TokenSequenceAndScalars[] | null;
  expectedNumSequences: number;
};

const HeatmapGrid: React.FC<HeatmapGridProps> = ({ tokenSequences, expectedNumSequences }) => {
  if (tokenSequences === null) {
    // No tokens specified means that we're rendering a skeleton without any content in it. The
    // width and minHeight specified below ensure that the skeleton is the same size as the actual
    // heatmap grid. We specify an array of nulls here, which the TokenHeatmap component will
    // handle gracefully.
    tokenSequences = new Array(expectedNumSequences).fill(null);
  }
  return (
    <div className="w-screen relative mt-6" style={{ marginLeft: "-50vw", left: "50%" }}>
      <div className="flex flow-row px-10 flex-wrap justify-center align-self-center">
        {tokenSequences.map((tokenSequence, i) => (
          <div
            className="my-3 border p-3 m-2 rounded-md"
            style={{ width: 400, minHeight: 194 }}
            key={i}
          >
            <TokenHeatmap tokenSequence={tokenSequence} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default HeatmapGrid;
