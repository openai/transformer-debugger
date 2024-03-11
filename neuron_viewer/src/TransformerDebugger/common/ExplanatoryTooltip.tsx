import React from "react";
import { Tooltip } from "@nextui-org/react";

// This component will result in its entire contents being tooltipped, such that hovering over any
// part of it will show the explanation.
//
// Usage:
//   <ExplanatoryTooltip explanation="Clear explanation of what's being shown">
//     <h3>Some contents</h3>
//   </ExplanatoryTooltip>
export const ExplanatoryTooltip: React.FC<{
  explanation: string;
  children: React.ReactNode;
}> = ({ explanation, children }) => {
  return (
    <Tooltip content={explanation} className="inline-block">
      {children}
    </Tooltip>
  );
};
