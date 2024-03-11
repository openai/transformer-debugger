export type Color = { r: number; g: number; b: number };

export function interpolateColor(colorLeft: Color, colorRight: Color, value: number): Color {
  const color = {
    r: Math.round(colorLeft.r + (colorRight.r - colorLeft.r) * value),
    g: Math.round(colorLeft.g + (colorRight.g - colorLeft.g) * value),
    b: Math.round(colorLeft.b + (colorRight.b - colorLeft.b) * value),
  };
  return color;
}

export function getInterpolatedColor(colors: Color[], boundaries: number[], value: number): Color {
  const index = boundaries.findIndex((boundary) => boundary >= value);
  const colorIndex = Math.max(0, index - 1);
  const colorLeft = colors[colorIndex];
  const colorRight = colors[colorIndex + 1];
  const boundaryLeft = boundaries[colorIndex];
  const boundaryRight = boundaries[colorIndex + 1];
  const ratio = (value - boundaryLeft) / (boundaryRight - boundaryLeft);
  const color = interpolateColor(colorLeft, colorRight, ratio);
  return color;
}

export const BLANK_COLOR: Color = { r: 255, g: 255, b: 255 }; // white
export const MAX_OUT_COLOR: Color = { r: 0, g: 255, b: 255 }; // cyan
export const MAX_IN_COLOR: Color = { r: 255, g: 0, b: 255 }; // magenta

export function subtractiveMix(color1: Color, color2: Color) {
  // Invert the colors
  let inverted1 = { r: 255 - color1.r, g: 255 - color1.g, b: 255 - color1.b };
  let inverted2 = { r: 255 - color2.r, g: 255 - color2.g, b: 255 - color2.b };

  // Mix them additively
  let mixed = {
    r: Math.min(inverted1.r + inverted2.r, 255),
    g: Math.min(inverted1.g + inverted2.g, 255),
    b: Math.min(inverted1.b + inverted2.b, 255),
  };

  // Invert the result
  return { r: 255 - mixed.r, g: 255 - mixed.g, b: 255 - mixed.b };
}

export const DEFAULT_BOUNDARIES = [0, 1];

export const DEFAULT_COLORS: Color[] = [
  { r: 255, g: 255, b: 255 },
  { r: 0, g: 255, b: 0 },
];

export const POSITIVE_NEGATIVE_COLORS: Color[] = [
  { r: 255, g: 0, b: 105 },
  { r: 255, g: 255, b: 255 },
  { r: 0, g: 255, b: 0 },
];

export const POSITIVE_NEGATIVE_BOUNDARIES = [0, 0.5, 1];
