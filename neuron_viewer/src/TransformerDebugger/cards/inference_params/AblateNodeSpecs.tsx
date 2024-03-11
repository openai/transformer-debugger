import {
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Button,
  Divider,
} from "@nextui-org/react";
import { makeNodeName } from "../../utils/nodes";
import { NodeType, InferenceAndTokenData } from "../../../client";
import { TokenLabel } from "./TokenLabel";
import { PromptInferenceParams } from "./inferenceParams";

type AblateNodeSpecsProps = {
  leftPromptInferenceParams: PromptInferenceParams;
  setLeftPromptInferenceParams: React.Dispatch<React.SetStateAction<PromptInferenceParams | null>>;
  inferenceAndTokenData: InferenceAndTokenData | null;
  twoPromptsMode: boolean;
};
export const AblateNodeSpecs: React.FC<AblateNodeSpecsProps> = ({
  leftPromptInferenceParams,
  setLeftPromptInferenceParams,
  inferenceAndTokenData,
  twoPromptsMode,
}) => {
  const nodeAblations = leftPromptInferenceParams.nodeAblations;
  if (nodeAblations.length === 0) {
    return null;
  }
  return (
    <div>
      <div className="flex flex-row gap-2 items-center">
        <span className="text-xl font-bold">Active ablations</span>
      </div>
      <div>
        {twoPromptsMode ? (
          <span>Ablations affect both prompts</span>
        ) : (
          <span>Left pane shows the ablated version; right pane shows the baseline version.</span>
        )}
      </div>
      <Table isStriped aria-label="Ablations" fullWidth={false}>
        <TableHeader>
          <TableColumn>Name</TableColumn>
          <TableColumn>Pass type</TableColumn>
          <TableColumn>Token attended to</TableColumn>
          <TableColumn>Token attended from</TableColumn>
          <TableColumn>Ablated to value</TableColumn>
          <TableColumn>Remove</TableColumn>
        </TableHeader>
        <TableBody>
          {nodeAblations.map((spec, index) => (
            <TableRow key={index}>
              <TableCell>{makeNodeName(spec.nodeIndex)}</TableCell>
              <TableCell>{spec.nodeIndex.passType}</TableCell>
              <TableCell>
                {spec.nodeIndex.nodeType === NodeType.ATTENTION_HEAD ? (
                  <TokenLabel
                    index={spec.nodeIndex.tensorIndices[1]}
                    inferenceAndTokenData={inferenceAndTokenData}
                  />
                ) : (
                  ""
                )}
              </TableCell>
              <TableCell>
                <TokenLabel
                  index={spec.nodeIndex.tensorIndices[0]}
                  inferenceAndTokenData={inferenceAndTokenData}
                />
              </TableCell>
              <TableCell>{spec.value}</TableCell>
              <TableCell>
                <Button
                  onClick={(e) => {
                    e.preventDefault();
                    const newAblations = [...nodeAblations];
                    newAblations.splice(index, 1);
                    setLeftPromptInferenceParams({
                      ...leftPromptInferenceParams,
                      nodeAblations: newAblations,
                    });
                  }}
                >
                  Remove
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Divider />
    </div>
  );
};
