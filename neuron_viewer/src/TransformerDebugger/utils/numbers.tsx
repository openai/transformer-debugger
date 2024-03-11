import { IRowNode } from "ag-grid-community";

export const formatFloat = (value: any, numDecimalPlaces: number = 2) => {
  return value !== undefined ? parseFloat(value).toFixed(numDecimalPlaces) : "";
};

export const formatFloatWithZeroPoint = (
  value: any,
  zeroPoint: number,
  numDecimalPlaces: number = 2
) => {
  return value !== undefined ? (parseFloat(value) - zeroPoint).toFixed(numDecimalPlaces) : "";
};

export const diffOptionalNumbers = (a: number | undefined, b: number | undefined) => {
  if (a === undefined) {
    a = 0;
  }
  if (b === undefined) {
    b = 0;
  }
  return a - b;
};

export const compareWithUndefinedAsZero = (
  a: number | undefined,
  b: number | undefined,
  unusedNodeA: IRowNode,
  unusedNodeB: IRowNode,
  // The grid itself handles inverting the order, so the comparator doesn't need to use it.
  unusedIsDescending: boolean
) => {
  if (a === undefined) {
    a = 0;
  }
  if (b === undefined) {
    b = 0;
  }
  return a - b;
};

export const compareWithUndefinedLast = (
  a: number | undefined,
  b: number | undefined,
  unusedNodeA: IRowNode,
  unusedNodeB: IRowNode,
  isDescending: boolean
) => {
  if (a === undefined) {
    a = isDescending ? -Infinity : Infinity;
  }
  if (b === undefined) {
    b = isDescending ? -Infinity : Infinity;
  }
  return a - b;
};
