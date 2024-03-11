import React, { useState, useMemo, useCallback, useEffect } from "react";
import { useParams, useLocation } from "react-router-dom";
import ModelInteractions from "./modelInteractions";
import { PaneComponents, PaneComponentType } from "./panes";
import {
  DimensionalityOfActivations,
  Node,
  PROMPTS_SEPARATOR,
  PaneData,
  getDimensionalityOfActivations,
  stringToNodeType,
} from "./types";
import { SectionTitle } from "./commonUiComponents";
import Navigation from "./navigation";
import { AttentionHeadRecordResponse, NeuronRecordResponse } from "./client";
import HeatmapGrid, { HeatmapGridProps } from "./heatmapGrid";
import HeatmapGrid2d, { HeatmapGrid2dProps } from "./heatmapGrid2d";
import { readAttentionHeadRecord, readNeuronRecord } from "./requests/readRequests";

// Math.random returns a value on [0, 1). Start each substring at index 2 to drop the "0." prefix.
const uuid = () =>
  Math.random().toString(36).substring(2) + Math.random().toString(36).substring(2);

const Pane: React.FC<{
  children: React.ReactNode;
  id?: string;
}> = ({ children, id }) => (
  <div id={id}>
    <div className="flex flex-col h-full">{children}</div>
  </div>
);

type PaneItem = { type: PaneComponentType; [key: string]: any };

function getHeatmapComponent(
  dimensionalityOfActivations: DimensionalityOfActivations
): React.FC<HeatmapGridProps> | React.FC<HeatmapGrid2dProps> {
  switch (dimensionalityOfActivations) {
    case DimensionalityOfActivations.SCALAR_PER_TOKEN:
      return HeatmapGrid;
    case DimensionalityOfActivations.SCALAR_PER_TOKEN_PAIR:
      return HeatmapGrid2d;
    default:
      throw new Error("Unknown dimensionality of activations: " + dimensionalityOfActivations);
  }
}

const NodePage = () => {
  const [additionalPanes, setAdditionalPanes] = useState<PaneData[]>([]);
  const [record, setRecord] = useState<NeuronRecordResponse | AttentionHeadRecordResponse | null>(
    null
  );
  const [isLoadingRecord, setIsLoadingRecord] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  let params =
    useParams<{ model: string; nodeTypeStr: string; layerIndex: string; nodeIndex: string }>();
  const activeNode: Node = useMemo(() => {
    if (
      params.nodeTypeStr === undefined ||
      params.layerIndex === undefined ||
      params.nodeIndex === undefined
    ) {
      throw new Error("One or more of node type, layer index, and node index are undefined.");
    } else {
      return {
        nodeType: stringToNodeType(params.nodeTypeStr),
        layerIndex: parseInt(params.layerIndex),
        nodeIndex: parseInt(params.nodeIndex),
      };
    }
  }, [params]);
  const dimensionalityOfActivations = getDimensionalityOfActivations(activeNode.nodeType);
  // Define this constant to make the checks below more concise and readable.
  const scalarPerToken =
    dimensionalityOfActivations === DimensionalityOfActivations.SCALAR_PER_TOKEN;

  const fetchRecord = useCallback(async () => {
    switch (dimensionalityOfActivations) {
      case DimensionalityOfActivations.SCALAR_PER_TOKEN:
        return await readNeuronRecord(activeNode);
      case DimensionalityOfActivations.SCALAR_PER_TOKEN_PAIR:
        return await readAttentionHeadRecord(activeNode);
      default:
        throw new Error("Unknown dimensionality of activations: " + dimensionalityOfActivations);
    }
  }, [activeNode, dimensionalityOfActivations]);

  useEffect(() => {
    async function fetchData() {
      try {
        const result = await fetchRecord();
        setErrorMessage(null);
        setRecord(result);
        setIsLoadingRecord(false);
      } catch (error) {
        if (error instanceof Error) {
          setErrorMessage(error.message);
        } else {
          setErrorMessage("Unknown error");
        }
      }
    }

    fetchData();
  }, [fetchRecord]);

  const addPane = useCallback((items: PaneItem[]) => {
    const newPanes = items.map((item) => {
      return {
        id: uuid().toString(),
        ...item,
      };
    });

    setAdditionalPanes((panes) => [...panes, ...newPanes]);
  }, []);

  const location = useLocation();
  const urlSearchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const promptsParam = urlSearchParams.get("promptsOfInterest");
  var promptsOfInterest: string[];
  if (promptsParam == null) {
    promptsOfInterest = [];
  } else {
    promptsOfInterest = promptsParam.split(PROMPTS_SEPARATOR);
  }

  return (
    <div>
      <Navigation activeNode={activeNode} />

      <div className="flow-root" style={{ width: 600, margin: "auto", overflow: "visible" }}>
        <ul className="mb-8 mt-10">
          {promptsOfInterest?.length ? (
            <>
              <SectionTitle>Prompts of interest</SectionTitle>
              <Pane>
                {promptsOfInterest.map((prompt, index) => (
                  <Pane key={index}>
                    <PaneComponents.ActivationsForPrompt
                      activeNode={activeNode}
                      sentence={prompt}
                    />
                  </Pane>
                ))}
              </Pane>
              <div style={{ marginBottom: "15px" }}>&nbsp;</div>
            </>
          ) : null}
          {
            <Pane>
              <PaneComponents.Explanation activeNode={activeNode} />
            </Pane>
          }
          <Pane>
            <PaneComponents.DatasetExamples
              record={record}
              isLoadingRecord={isLoadingRecord}
              errorMessage={errorMessage}
              heatmapComponent={getHeatmapComponent(dimensionalityOfActivations)}
            />
          </Pane>
          <Pane>
            <PaneComponents.LogitLens activeNode={activeNode} />
          </Pane>

          {additionalPanes.map((pane) => (
            <li key={pane.id}>
              <div className="relative">
                <div className="relative flex items-start space-x-3">
                  <Pane key={pane.id} id={pane.id} {...pane}>
                    {React.createElement(PaneComponents[pane.type] as any, {
                      ...pane,
                      key: pane.id,
                      activeNode: activeNode!,
                    })}
                  </Pane>
                </div>
              </div>
            </li>
          ))}
        </ul>
        <div>
          <ModelInteractions
            onGetActivationsForPrompt={(sentence) => {
              addPane([{ type: "ActivationsForPrompt", sentence }]);
            }}
            // Scoring explanations is only supported for neurons and autoencoder latents.
            onScoreExplanation={
              scalarPerToken
                ? (explanation) => {
                    addPane([{ type: "ScoreExplanation", explanation }]);
                  }
                : undefined
            }
          />
        </div>
      </div>
    </div>
  );
};

export default NodePage;
