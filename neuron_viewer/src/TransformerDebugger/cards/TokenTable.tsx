import React from "react";
import { formatToken } from "../../tokenRendering";

type TokenTableProps = {
  leftTokens: string[];
  rightTokens?: string[];
};

const TokenTable: React.FC<TokenTableProps> = ({ leftTokens, rightTokens }) => {
  const tableStyle: React.CSSProperties = {
    maxWidth: "800px",
    margin: "0 auto",
    borderCollapse: "collapse",
    borderColor: "#f0f0f0",
  };

  const cellStyle: React.CSSProperties = {
    textAlign: "center",
    padding: "5px",
    border: "1px solid #f0f0f0",
    fontFamily: "monospace",
  };

  const indexMismatchStyle: React.CSSProperties = {
    ...cellStyle,
    backgroundColor: "#ffcccc", // Light red background for mismatched indices
  };

  const rowNameStyle: React.CSSProperties = {
    ...cellStyle,
    fontWeight: "bold",
    fontFamily: "sans-serif",
  };

  const isMismatchAtIndex = (index: number) => {
    return rightTokens && leftTokens[index] !== rightTokens[index];
  };

  return (
    <table className="token-table" style={tableStyle}>
      <tbody>
        <tr>
          <td style={rowNameStyle}>{rightTokens ? "Left token" : "Token"}</td>
          {leftTokens.map((token, i) => (
            <td key={`left-token-${i}`} style={cellStyle}>
              {formatToken(token)}
            </td>
          ))}
        </tr>
        {rightTokens && (
          <tr>
            <td style={rowNameStyle}>Right token</td>
            {rightTokens.map((token, i) => (
              <td key={`right-token-${i}`} style={cellStyle}>
                {formatToken(token)}
              </td>
            ))}
          </tr>
        )}
        <tr>
          <td style={rowNameStyle}>Index</td>
          {leftTokens.map((_, i) => (
            <td key={`index-${i}`} style={isMismatchAtIndex(i) ? indexMismatchStyle : cellStyle}>
              {i}
            </td>
          ))}
        </tr>
      </tbody>
    </table>
  );
};

export default TokenTable;
