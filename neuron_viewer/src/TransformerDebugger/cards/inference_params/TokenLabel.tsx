// Displays a token index and optional token string.
import React from "react";
import { InferenceAndTokenData } from "../../../client";
import { renderToken } from "../../../tokenRendering";

export const TokenLabel: React.FC<{
  index: number;
  tokenString?: string;
  inferenceAndTokenData: InferenceAndTokenData | null;
}> = ({ index, tokenString, inferenceAndTokenData }) => {
  const currentTokenString = inferenceAndTokenData?.tokensAsStrings[index] || tokenString || "";
  return (
    <>
      {renderToken(currentTokenString)} ({index})
    </>
  );
};
