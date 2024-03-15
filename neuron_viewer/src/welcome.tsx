import { useState, FormEvent, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { dstStringToNodeType } from "./types";
import { NeuronDatasetMetadata } from "./client";
import { readNeuronDatasetsMetadata } from "./requests/readRequests";
import { getModelInfo } from "./requests/inferenceRequests";

const Welcome: React.FC = () => {
  // Keys are dataset keys returned by getDatasetKey; values are user-written strings of the form
  // "{layerIndex}:{nodeIndex}".
  const [inputValues, setInputValues] = useState<{ [dataset: string]: string }>({});
  const [datasetsMetadata, setDatasetsMetadata] = useState<NeuronDatasetMetadata[] | null>(null);
  const [tdbUrl, setTdbUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    async function fetchData() {
      try {
        const result = await readNeuronDatasetsMetadata();
        setErrorMessage(null);
        setDatasetsMetadata(result);
        setIsLoading(false);
      } catch (error) {
        if (error instanceof Error) {
          setErrorMessage(error.message);
        } else {
          setErrorMessage("Unknown error");
        }
      }
      try {
        const modelInfo = await getModelInfo();
        if (modelInfo.modelName ?? false) {
          if (modelInfo.hasMlpAutoencoder && modelInfo.hasAttentionAutoencoder) {
            setTdbUrl(`/${modelInfo.modelName}_${modelInfo.mlpAutoencoderName}_${modelInfo.attentionAutoencoderName}/tdb_alpha`);
          } else if (modelInfo.hasMlpAutoencoder) {
            setTdbUrl(`/${modelInfo.modelName}_${modelInfo.mlpAutoencoderName}/tdb_alpha`);
          } else if (modelInfo.hasAttentionAutoencoder) {
            setTdbUrl(`/${modelInfo.modelName}_${modelInfo.attentionAutoencoderName}/tdb_alpha`);
          } else {
            setTdbUrl(`/${modelInfo.modelName}/tdb_alpha`);
          }
        }
      } catch (error) {
        // Continue without TDB link, it just won't be displayed
        console.error("Failed to get model info", error);
      }
    }

    fetchData();
  }, []);

  const handleSubmit = (datasetKey: string, e: FormEvent) => {
    e.preventDefault();
    const [layerIndex, nodeIndex] = inputValues[datasetKey].split(":");
    if (isNaN(Number(layerIndex)) || isNaN(Number(nodeIndex))) {
      throw new Error("Invalid input: expected {layerIndex}:{nodeIndex}");
    }
    navigate(`/${datasetKey}/${layerIndex}/${nodeIndex}`);
  };

  const getDatasetKey = (datasetMetadata: NeuronDatasetMetadata) => {
    return datasetMetadata.shortName + "/" + dstStringToNodeType(datasetMetadata.derivedScalarType);
  };

  return (
    <div>
      <div className="flow-root w-[600px] mx-auto overflow-visible mb-8">
        <ul className="mb-8 mt-10">
          <div className="flex flex-col items-center justify-center">
            <h1 className="text-4xl font-bold mb-4 w-[400px]">Neuron viewer</h1>
            <div className="mt-4">
              {isLoading ? (
                <div className="flex justify-center items-center">
                  <p className="text-gray-500 mb-2">Loading...</p>
                </div>
              ) : errorMessage != null ? (
                <div className="flex justify-center items-center">
                  <p className="text-gray-500 mb-2">
                    Failed to load dataset metadata: {errorMessage}
                  </p>
                </div>
              ) : (
                <>
                  {datasetsMetadata!.map((datasetMetadata) => (
                    <div key={getDatasetKey(datasetMetadata)} className="mb-2">
                      <a href={`/${getDatasetKey(datasetMetadata)}/0/0`}>
                        {datasetMetadata.userVisibleName}
                      </a>
                      <form
                        onSubmit={(e) => handleSubmit(getDatasetKey(datasetMetadata), e)}
                        className="flex flex-col items-center justify-center"
                      >
                        <input
                          type="text"
                          id={getDatasetKey(datasetMetadata)}
                          value={inputValues[getDatasetKey(datasetMetadata)]}
                          onChange={(e) => {
                            setInputValues({
                              ...inputValues,
                              [getDatasetKey(datasetMetadata)]: e.target.value,
                            });
                          }}
                          placeholder="{layerIndex}:{nodeIndex}"
                          className="border border-gray-300 rounded-md p-2 mb-4 w-[400px]"
                        />
                      </form>
                    </div>
                  ))}
                </>
              )}
            </div>
            {tdbUrl && (
              <a href={tdbUrl} className="text-4xl font-bold mb-4 w-[400px]">Link to Transformer Debugger</a>
            )}
          </div>
        </ul>
      </div>
    </div>
  );
};

export default Welcome;
