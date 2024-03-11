import { Button, Input } from "@nextui-org/react";

export const MultiTokenInput: React.FC<{
  tokens: string[];
  onChange: (tokens: string[]) => void;
  className?: string;
  allowLengthZero?: boolean;
}> = ({ tokens, onChange, className, allowLengthZero }) => {
  // display a row of text inputs with one token per input, + button to add more tokens, - button to remove last token
  // when token is changed, call onChange with new tokens
  const allowRemovingTokens = tokens.length > 1 || (allowLengthZero && tokens.length === 1);

  return (
    <div className={`flex flex-row gap-2 ${className}`}>
      {tokens.map((token, index) => (
        <Input
          className="pt-0 pb-0 sm:text-sm block mr-0 w-auto font-mono"
          size="sm"
          key={index}
          type="text"
          value={token}
          onValueChange={(value) => {
            const newTokens = [...tokens];
            newTokens[index] = value;
            onChange(newTokens);
          }}
        />
      ))}
      <Button
        onClick={(e) => {
          e.preventDefault();
          onChange([...tokens, ""]);
        }}
      >
        Add token
      </Button>
      <Button
        className="disabled:opacity-50"
        disabled={!allowRemovingTokens}
        onClick={(e) => {
          e.preventDefault();
          if (!allowRemovingTokens) {
            return;
          }
          onChange(tokens.slice(0, tokens.length - 1));
        }}
      >
        Remove token
      </Button>
    </div>
  );
};
