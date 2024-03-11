// This file contains a component that can render both neuron and attention head dataset examples.

import { TokenSequenceAndScalars, TokenSequenceAndAttentionScalars } from "../types";
import HeatmapGrid from "../heatmapGrid";
import HeatmapGrid2d from "../heatmapGrid2d";
import { SectionTitle, ShowAllOrFewerButton } from "../commonUiComponents";
import { useState } from "react";
import { AttentionHeadRecordResponse, NeuronRecordResponse } from "../client";

const SubSectionTitle: React.FC<{ title: string }> = ({ title }) => (
  <h3 className="text-xl font-bold" style={{ padding: "0" }}>
    {title}
  </h3>
);

function extractTokenData(record: NeuronRecordResponse | AttentionHeadRecordResponse) {
  // note: although the type is TokenSequenceAnd(Attention)Scalars, the scalars are what we call 'activations'
  // since they come from a NeuronRecord object
  let topTokensWithActivations: Array<TokenSequenceAndScalars | TokenSequenceAndAttentionScalars> =
    [];
  let randomTokensWithActivations: Array<
    TokenSequenceAndScalars | TokenSequenceAndAttentionScalars
  > = [];

  if ("topActivations" in record && "randomSample" in record) {
    // It's a NeuronRecordResponse
    topTokensWithActivations = record.topActivations;
    randomTokensWithActivations = record.randomSample;
  } else if ("mostPositiveTokenSequences" in record && "randomTokenSequences" in record) {
    // It's an AttentionHeadRecordResponse
    topTokensWithActivations = record.mostPositiveTokenSequences;
    randomTokensWithActivations = record.randomTokenSequences;
  }

  return { topTokensWithActivations, randomTokensWithActivations };
}

type HeatmapComponentType = typeof HeatmapGrid | typeof HeatmapGrid2d;

interface RenderHeatmapProps {
  heatmapComponent: HeatmapComponentType;
  shownSequences: Array<TokenSequenceAndScalars | TokenSequenceAndAttentionScalars>;
  numToShow: number;
}

const renderHeatmap = ({
  heatmapComponent,
  shownSequences,
  numToShow,
}: RenderHeatmapProps): JSX.Element => {
  if (heatmapComponent === HeatmapGrid) {
    return (
      <HeatmapGrid
        tokenSequences={shownSequences as Array<TokenSequenceAndScalars>}
        expectedNumSequences={numToShow}
      />
    );
  } else if (heatmapComponent === HeatmapGrid2d) {
    return (
      <HeatmapGrid2d
        tokenSequenceAndAttentionScalars={shownSequences as Array<TokenSequenceAndAttentionScalars>}
        expectedNumSequences={numToShow}
      />
    );
  }

  throw new Error("Invalid heatmap component");
};

type DatasetExamplesProps<T> = {
  record: T | null;
  isLoadingRecord: boolean;
  errorMessage: string | null;
  heatmapComponent: HeatmapComponentType;
};

const DatasetExamples: React.FC<
  DatasetExamplesProps<NeuronRecordResponse | AttentionHeadRecordResponse>
> = ({ record, isLoadingRecord, errorMessage, heatmapComponent }) => {
  const [showAll, setShowAll] = useState(false);

  if (errorMessage != null) {
    return (
      <div className="flex justify-center items-center">
        <p className="text-gray-500 mb-2">Failed to load data: {errorMessage}</p>
      </div>
    );
  }

  const numTopToShow = 6;
  const numRandomToShow = 3;
  let shownTopSequences, shownRandomSequences;
  if (isLoadingRecord) {
    shownTopSequences = null;
    shownRandomSequences = null;
  } else {
    if (!record) {
      throw new Error("Record is null or undefined");
    }
    const { topTokensWithActivations, randomTokensWithActivations } = extractTokenData(record);
    shownTopSequences = showAll
      ? topTokensWithActivations
      : topTokensWithActivations.slice(0, numTopToShow);
    shownRandomSequences = showAll
      ? randomTokensWithActivations
      : randomTokensWithActivations.slice(0, numRandomToShow);
  }

  return (
    <div>
      <SectionTitle>Dataset examples</SectionTitle>
      <SubSectionTitle title="Top" />
      {renderHeatmap({
        heatmapComponent,
        shownSequences: shownTopSequences!,
        numToShow: numTopToShow,
      })}
      <ShowAllOrFewerButton showAll={showAll} setShowAll={setShowAll} />
      <div className="h-8"></div>
      <hr />
      <div className="h-8"></div>
      <SubSectionTitle title="Random" />
      {renderHeatmap({
        heatmapComponent,
        shownSequences: shownRandomSequences!,
        numToShow: numRandomToShow,
      })}
      <ShowAllOrFewerButton showAll={showAll} setShowAll={setShowAll} />
      <div className="h-8"></div>
    </div>
  );
};

export default DatasetExamples;
