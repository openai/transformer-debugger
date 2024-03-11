import { ReactNode } from "react";

export const SectionTitle = ({ children }: { children: ReactNode }) => {
  return <h2 className="text-3xl font-bold mb-4">{children}</h2>;
};

export const defaultSmallButtonClasses =
  "text-black no-underline text-base border-black border font-sans bg-white font-small inline-block rounded " +
  "transition-all duration-200 ease-in-out hover:bg-gray-100 disabled:bg-gray-300 disabled:cursor-not-allowed px-1 py-0";

export const ShowAllOrFewerButton = ({
  showAll,
  setShowAll,
}: {
  showAll: boolean;
  setShowAll: (showAll: boolean) => void;
}) => {
  return (
    <button className={defaultSmallButtonClasses} onClick={() => setShowAll(!showAll)}>
      {showAll ? "Show fewer" : "Show all"}
    </button>
  );
};
