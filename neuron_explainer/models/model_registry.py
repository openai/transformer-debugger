from dataclasses import dataclass

import torch

from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.models import Transformer
from neuron_explainer.models.autoencoder_context import (
    AutoencoderConfig,
    AutoencoderContext,
    AutoencoderSpec,
)


@dataclass(frozen=True)
class StandardModelSpec:
    model_path: str  # checkpoint path


_MODEL_SPECS: dict[str, StandardModelSpec] = {
    # GPT-2 series
    "gpt2-small": StandardModelSpec(
        model_path="az://openaipublic/neuron-explainer/subject-models/gpt2/small"
    ),
    "gpt2-medium": StandardModelSpec(
        model_path="az://openaipublic/neuron-explainer/subject-models/gpt2/medium"
    ),
    "gpt2-large": StandardModelSpec(
        model_path="az://openaipublic/neuron-explainer/subject-models/gpt2/large"
    ),
    "gpt2-xl": StandardModelSpec(
        model_path="az://openaipublic/neuron-explainer/subject-models/gpt2/xl"
    ),
}

_AUTOENCODER_SPECS: dict[str, dict[str, AutoencoderSpec]] = {
    "gpt2-small": {
        # released December 2023
        "ae-mlp-post-act-v1": AutoencoderSpec(
            dst=DerivedScalarType.MLP_POST_ACT,
            autoencoder_path_by_layer_index={
                layer_index: f"az://openaipublic/sparse-autoencoder/gpt2-small/mlp_post_act/autoencoders/{layer_index}.pt"
                for layer_index in range(12)
            },
        ),
        "ae-resid-delta-mlp-v1": AutoencoderSpec(
            dst=DerivedScalarType.RESID_DELTA_MLP,
            autoencoder_path_by_layer_index={
                layer_index: f"az://openaipublic/sparse-autoencoder/gpt2-small/resid_delta_mlp/autoencoders/{layer_index}.pt"
                for layer_index in range(12)
            },
        ),
        # released March 2024
        "ae-mlp-post-act-v4": AutoencoderSpec(
            dst=DerivedScalarType.MLP_POST_ACT,
            autoencoder_path_by_layer_index={
                layer_index: f"az://openaipublic/sparse-autoencoder/gpt2-small/mlp_post_act_v4/autoencoders/{layer_index}.pt"
                for layer_index in range(12)
            },
        ),
        "ae-resid-delta-mlp-v4": AutoencoderSpec(
            dst=DerivedScalarType.RESID_DELTA_MLP,
            autoencoder_path_by_layer_index={
                layer_index: f"az://openaipublic/sparse-autoencoder/gpt2-small/resid_delta_mlp_v4/autoencoders/{layer_index}.pt"
                for layer_index in range(12)
            },
        ),
        "ae-resid-delta-attn-v4": AutoencoderSpec(
            dst=DerivedScalarType.RESID_DELTA_ATTN,
            autoencoder_path_by_layer_index={
                layer_index: f"az://openaipublic/sparse-autoencoder/gpt2-small/resid_delta_attn_v4/autoencoders/{layer_index}.pt"
                for layer_index in range(12)
            },
        ),
    },
}


def list_autoencoder_names(model_name: str = "gpt2-small") -> list[str]:
    return list(_AUTOENCODER_SPECS[model_name].keys())


def get_standard_model_spec(model_name: str) -> StandardModelSpec:
    return _MODEL_SPECS[model_name]


def load_standard_transformer(model_name: str, device: torch.device | None = None) -> Transformer:
    print(f"Loading standard model {model_name}...")
    model_spec = get_standard_model_spec(model_name)
    return load_standard_transformer_from_model_spec(model_spec, device=device)


def load_standard_transformer_from_model_spec(
    model_spec: StandardModelSpec, device: torch.device | None = None
) -> Transformer:
    return Transformer.load(
        model_spec.model_path,
        dtype=torch.float32,
        device=device,
    )


def make_autoencoder_context(
    model_name: str,
    autoencoder_name: str,
    device: torch.device,
    omit_dead_latents: bool = False,
) -> AutoencoderContext:
    try:
        autoencoder_spec = _AUTOENCODER_SPECS[model_name][autoencoder_name]
    except KeyError:
        raise ValueError(
            f"No autoencoder spec found for model {model_name} and autoencoder {autoencoder_name}. "
            f"Available autoencoders for model {model_name} are: {list(_AUTOENCODER_SPECS[model_name].keys())}"
        )
    autoencoder_config = AutoencoderConfig.from_spec(autoencoder_spec)
    autoencoder_context = AutoencoderContext(
        autoencoder_config=autoencoder_config,
        device=device,
        omit_dead_latents=omit_dead_latents,
    )
    return autoencoder_context
