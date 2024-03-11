import React, { useEffect, useState } from "react";

// fetchDataFunc and displayDataFunc should be created with useCallback, otherwise
// they could change every time the parent renders and trigger a rerun of the underlying request
export interface FetchAndDisplayProps<PanePropsT, DataT> {
  paneProps: PanePropsT;
  fetchDataFunc: () => Promise<DataT>;
  displayDataFunc: (
    data: DataT,
    isLoading: boolean,
    showAll: boolean,
    setShowAll: (showAll: boolean) => void
  ) => JSX.Element;
  initialData?: DataT;
}

export const FetchAndDisplayPane = <PanePropsT, DataT>({
  paneProps,
  fetchDataFunc,
  displayDataFunc,
  initialData,
}: FetchAndDisplayProps<PanePropsT, DataT>): React.ReactElement => {
  const [data, setData] = useState<DataT | null>(initialData ?? null);
  const [showAll, setShowAll] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const result = await fetchDataFunc();
        setErrorMessage(null);
        setData(result);
        setIsLoading(false);
      } catch (error) {
        if (error instanceof Error) {
          setErrorMessage(error.message);
        } else {
          setErrorMessage("Unknown error");
        }
      }
    }

    fetchData();
  }, [paneProps, fetchDataFunc]);

  if (errorMessage != null) {
    return (
      <div className="flex justify-center items-center">
        <p className="text-gray-500 mb-2">Failed to load data: {errorMessage}</p>
      </div>
    );
  }

  return displayDataFunc(data!, isLoading, showAll, setShowAll);
};
