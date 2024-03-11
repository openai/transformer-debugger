import { Checkbox } from "@nextui-org/react";
import { ExplanatoryTooltip } from "../common/ExplanatoryTooltip";

const displayKeys: { [key: string]: { label: string; explanation: string } } = {
  logits: {
    label: "Show logits display",
    explanation:
      "Whether to show a table listing top candidates for the next token with their logits.",
  },
  bySequenceToken: {
    label: "Show token effect display",
    explanation:
      "Whether to show the prompt, with each token colored by the estimated total effect summed over all nodes of a same type (MLP neurons, attention heads, embeddings).",
  },
  node: {
    label: "Show node table",
    explanation:
      "Whether to show a table of nodes (MLP neurons, attention heads, autoencoder latents, etc.) and their effect on the direction of interest.",
  },
};

const DisplayOptions = ({
  displaySettings,
  toggleDisplay,
}: {
  displaySettings: Map<string, boolean>;
  toggleDisplay: (key: string) => void;
}) => {
  return (
    <>
      {Object.keys(displayKeys).map((key) => (
        <ExplanatoryTooltip explanation={displayKeys[key].explanation} key={key}>
          <div>
            <Checkbox
              type="checkbox"
              isSelected={displaySettings.get(key)}
              onValueChange={() => toggleDisplay(key)}
            >
              {displayKeys[key].label}
            </Checkbox>
          </div>
        </ExplanatoryTooltip>
      ))}
    </>
  );
};

export default DisplayOptions;
