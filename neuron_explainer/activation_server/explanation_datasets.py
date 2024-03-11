import os

from neuron_explainer.activation_server.load_neurons import convert_dataset_path_to_short_name

# Maps from neuron dataset path to explanation dataset path.
AZURE_EXPLANATION_DATASET_REGISTRY = {
    "az://openaipublic/neuron-explainer/data/collated-activations/": "az://openaipublic/neuron-explainer/data/explanations/",
    "az://openaipublic/neuron-explainer/gpt2_small_data/collated-activations/": "az://openaipublic/neuron-explainer/gpt2_small_data/explanations/",
}


def get_local_cached_explanation_directory(dataset_path: str) -> str:
    root_project_directory = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dataset_short_name = convert_dataset_path_to_short_name(dataset_path)
    return f"{root_project_directory}/cached_explanations/{dataset_short_name}"


async def get_all_explanation_datasets(neuron_dataset: str) -> list[str]:
    """
    Get all explanation datasets for a given neuron dataset. Search the public azure bucket and also
    the local filesystem cache. Returns a list of paths to the explanation datasets.
    Path can be an azure path (beginning with `az://`) or a local path.
    """
    datasets = []
    if neuron_dataset in AZURE_EXPLANATION_DATASET_REGISTRY:
        datasets.append(AZURE_EXPLANATION_DATASET_REGISTRY[neuron_dataset])
    local_cache_dir = get_local_cached_explanation_directory(neuron_dataset)
    # Iterate through folders to get a list of dirs.
    # There will be different local cache directories if the user generates scored explanations for
    # the same neuron dataset using different neuron/attention explainer registry entries (i.e. so
    # that AttentionExplainAndScoreMethodId or NeuronExplainAndScoreMethodId differ).
    if os.path.exists(local_cache_dir) and os.path.isdir(local_cache_dir):
        for entry in os.listdir(local_cache_dir):
            candidate_path = os.path.join(local_cache_dir, entry)
            if os.path.isdir(candidate_path):
                datasets.append(candidate_path)
    return datasets
