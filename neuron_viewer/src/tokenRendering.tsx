// Functions for formatting and rendering tokens.

export function formatToken(token: string, dotsForSpaces: boolean = true) {
  const result = token.replace(/\n/g, "↵");
  if (dotsForSpaces) {
    // Note: There's a zero-width space just before the middle dot below, to allow line wrapping.
    return result.replace(/ /g, "​·");
  }
  return result;
}

export function renderToken(token: string) {
  return <span className="font-mono">{formatToken(token)}</span>;
}

export function renderTokenOnBlue(token: string, key?: number) {
  return renderTokenOnColor(token, "bg-blue-100", key);
}

export function renderTokenOnGray(token: string, key?: number) {
  return renderTokenOnColor(token, "bg-gray-100", key);
}

export function renderTokenOnColor(token: string, bgColorClass: string, key?: number) {
  return (
    <span
      key={key}
      className={`inline-flex m-1 items-center px-2.5 rounded-full text-xs font-medium ${bgColorClass} text-gray-800 font-mono`}
    >
      {formatToken(token)}
    </span>
  );
}
