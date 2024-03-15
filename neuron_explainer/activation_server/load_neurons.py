from fastapi import HTTPException

from neuron_explainer.activation_server.neuron_datasets import (
    NEURON_DATASET_METADATA_REGISTRY,
    get_neuron_dataset_metadata_by_short_name_and_dst,
)
from neuron_explainer.activations.activations import NeuronRecord, load_neuron_async
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.pydantic import CamelCaseBaseModel, immutable


@immutable
class NodeIdAndDatasets(CamelCaseBaseModel):
    dst: DerivedScalarType
    layer_index: int
    activation_index: int
    datasets: list[str]
    """A list of dataset paths or short names."""


def resolve_neuron_dataset(dataset: str, dst: DerivedScalarType) -> str:
    if dataset.startswith("https://"):
        return dataset
    else:
        # It's the short name for a dataset, like "gpt2-small". We have to look up the metadata.
        dataset_metadata = get_neuron_dataset_metadata_by_short_name_and_dst(dataset, dst)
        return dataset_metadata.neuron_dataset_path


def convert_dataset_path_to_short_name(dataset_path: str) -> str:
    assert dataset_path.startswith("https://")
    short_name = None
    for metadata in NEURON_DATASET_METADATA_REGISTRY.values():
        if metadata.neuron_dataset_path == dataset_path:
            short_name = metadata.short_name
            break
    assert (
        short_name is not None
    ), f"Could not find short name for {dataset_path}. If you're trying to use a custom dataset, ensure that you have added it to neuron_datasets.py:NEURON_DATASET_METADATA_REGISTRY."
    return short_name


async def load_neuron_from_datasets(
    node_id_and_datasets: NodeIdAndDatasets,
) -> tuple[str, NeuronRecord]:
    """
    Load a neuron record of the specified dst (e.g. DerivedScalarType.MLP_POST_ACT) from a list of
    datasets, returning the data from the first dataset that has the neuron.

    Used to allow first trying a dataset that only covers a subset of neurons for a model,
    with a fallback to another dataset that covers all neurons.
    """
    dst = node_id_and_datasets.dst
    datasets = node_id_and_datasets.datasets
    dataset_paths = [resolve_neuron_dataset(dataset, dst) for dataset in datasets]
    layer_index = node_id_and_datasets.layer_index
    activation_index = node_id_and_datasets.activation_index
    for dataset_path in dataset_paths:
        try:
            return dataset_path, await load_neuron_async(
                dataset_path, layer_index, activation_index
            )
        except FileNotFoundError:
            pass
    raise HTTPException(
        status_code=404,
        detail=f"Could not find {dst} {layer_index}:{activation_index} in {dataset_paths}",
    )
