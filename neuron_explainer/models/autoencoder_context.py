import os
from dataclasses import dataclass, field
from typing import Union

import blobfile as bf
import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.file_utils import file_exists
from neuron_explainer.models import Autoencoder
from neuron_explainer.models.model_component_registry import Dimension, LayerIndex, NodeType


@dataclass(frozen=True)
class AutoencoderSpec:
    """Parameters used in the construction of an AutoencoderConfig object. Seperate so we don't need to validate when constructed"""

    dst: DerivedScalarType
    autoencoder_path_by_layer_index: dict[LayerIndex, str]


@dataclass(frozen=True)
class AutoencoderConfig:
    """
    This class specifies a set of autoencoders to load from disk, for one or more layer indices.
    The activation location type indicates the type of activation that the autoencoder was trained
    on, and that will be fed into the autoencoder.
    """

    dst: DerivedScalarType
    autoencoder_path_by_layer_index: dict[LayerIndex, str]

    def __post_init__(self) -> None:
        assert len(self.autoencoder_path_by_layer_index) > 0
        if len(self.autoencoder_path_by_layer_index) > 1:
            assert (
                None not in self.autoencoder_path_by_layer_index.keys()
            ), "layer_indices must be [None], or a list of int layer indices"

    @classmethod
    def from_spec(cls, params: AutoencoderSpec) -> "AutoencoderConfig":
        return cls(
            dst=params.dst,
            autoencoder_path_by_layer_index=params.autoencoder_path_by_layer_index,
        )


@dataclass(frozen=True)
class AutoencoderContext:
    autoencoder_config: AutoencoderConfig
    device: torch.device
    _cached_autoencoders_by_path: dict[str, Autoencoder] = field(default_factory=dict)
    omit_dead_latents: bool = False
    """
    Omit dead latents to save memory. Only happens if self.warmup() is called. Because we omit the
    same number of latents from all autoencoders, we can only omit the smallest number of dead
    latents among all autoencoders.
    """

    @property
    def num_autoencoder_directions(self) -> int:
        """Note that this property might change after warmup() is called, if omit_dead_latents is True."""
        if len(self._cached_autoencoders_by_path) == 0:
            raise ValueError(
                "num_autoencoder_directions is not populated yet. Call warmup() first."
            )
        else:
            # all autoencoders have the same number of directions, so we can just check one
            first_autoencoder = next(iter(self._cached_autoencoders_by_path.values()))
            return first_autoencoder.latent_bias.shape[0]

    @property
    def _min_n_dead_latents(self) -> int:
        return min(
            count_dead_latents(autoencoder)
            for autoencoder in self._cached_autoencoders_by_path.values()
        )

    @property
    def dst(self) -> DerivedScalarType:
        return self.autoencoder_config.dst

    @property
    def layer_indices(self) -> set[LayerIndex]:
        return set(self.autoencoder_config.autoencoder_path_by_layer_index.keys())

    def get_autoencoder(self, layer_index: LayerIndex) -> Autoencoder:
        autoencoder_azure_path = self.autoencoder_config.autoencoder_path_by_layer_index.get(
            layer_index
        )
        if autoencoder_azure_path is None:
            raise ValueError(f"No autoencoder path for layer_index {layer_index}")
        else:
            if autoencoder_azure_path in self._cached_autoencoders_by_path:
                autoencoder = self._cached_autoencoders_by_path[autoencoder_azure_path]
            else:
                # Check if the autoencoder is cached on disk
                disk_cache_path = os.path.join(
                    "/tmp", autoencoder_azure_path.replace("https://", "")
                )
                os.makedirs(os.path.dirname(disk_cache_path), exist_ok=True)
                if file_exists(disk_cache_path):
                    print(f"Loading autoencoder from disk cache: {disk_cache_path}")
                else:
                    print(f"Reading autoencoder from blob storage: {autoencoder_azure_path}")
                    # Cache the autoencoder to disk, using bf.copy to make sure md5 is preserved
                    bf.copy(autoencoder_azure_path, disk_cache_path, overwrite=True)

                state_dict = torch.load(disk_cache_path, map_location=self.device)
                # released autoencoders are saved as a dict for better compatibility
                assert isinstance(state_dict, dict)
                autoencoder = Autoencoder.from_state_dict(state_dict, strict=False).to(self.device)
                self._cached_autoencoders_by_path[autoencoder_azure_path] = autoencoder

            # freeze the autoencoder
            for p in autoencoder.parameters():
                p.requires_grad = False
            return autoencoder

    def warmup(self) -> None:
        """Load all autoencoders into memory."""
        for layer_index in self.layer_indices:
            self.get_autoencoder(layer_index)

        # num_autoencoder_directions is always populated after warmup
        n_latents = self.num_autoencoder_directions

        if self.omit_dead_latents:
            # drop the dead latents to save memory, but keep the same number of directions for all autoencoders
            if self._min_n_dead_latents > 0:
                print(f"Omitting {self._min_n_dead_latents} dead latents from all autoencoders")
                n_latents_to_keep = n_latents - self._min_n_dead_latents
                for key, autoencoder in self._cached_autoencoders_by_path.items():
                    self._cached_autoencoders_by_path[key] = omit_least_active_latents(
                        autoencoder, n_latents_to_keep=n_latents_to_keep
                    )

    def get_parameterized_dimension_sizes(self) -> dict[Dimension, int]:
        """A dictionary specifying the size of the parameterized dimensions; for convenient use with ScalarDerivers"""
        return {
            Dimension.AUTOENCODER_LATENTS: self.num_autoencoder_directions,
        }

    @property
    def autoencoder_node_type(self) -> NodeType | None:
        return _autoencoder_node_type_by_input_dst.get(self.dst)


_autoencoder_node_type_by_input_dst = {
    # add more mappings as needed
    DerivedScalarType.MLP_POST_ACT: NodeType.MLP_AUTOENCODER_LATENT,
    DerivedScalarType.RESID_DELTA_MLP_FROM_MLP_POST_ACT: NodeType.MLP_AUTOENCODER_LATENT,
    DerivedScalarType.RESID_DELTA_MLP: NodeType.MLP_AUTOENCODER_LATENT,
    DerivedScalarType.RESID_DELTA_ATTN: NodeType.ATTENTION_AUTOENCODER_LATENT,
    DerivedScalarType.ATTN_WRITE: NodeType.ATTENTION_AUTOENCODER_LATENT,
}


@dataclass(frozen=True)
class MultiAutoencoderContext:
    autoencoder_context_by_node_type: dict[NodeType, AutoencoderContext]

    @classmethod
    def from_context_or_multi_context(
        cls,
        input: Union[AutoencoderContext, "MultiAutoencoderContext", None],
    ) -> Union["MultiAutoencoderContext", None]:
        if isinstance(input, AutoencoderContext):
            return cls.from_autoencoder_context_list([input])
        elif input is None:
            return None
        else:
            return input

    @classmethod
    def from_autoencoder_context_list(
        cls, autoencoder_context_list: list[AutoencoderContext]
    ) -> "MultiAutoencoderContext":
        # check if there are duplicate node types
        node_types = [
            _autoencoder_node_type_by_input_dst[autoencoder_context.dst]
            for autoencoder_context in autoencoder_context_list
        ]
        if len(node_types) != len(set(node_types)):
            raise ValueError(f"Cannot load two autoencoders with the same node type ({node_types})")
        return cls(
            autoencoder_context_by_node_type={
                _autoencoder_node_type_by_input_dst[autoencoder_context.dst]: autoencoder_context
                for autoencoder_context in autoencoder_context_list
            }
        )

    def get_autoencoder_context(
        self, node_type: NodeType | None = None
    ) -> AutoencoderContext | None:
        if node_type is None or node_type == NodeType.AUTOENCODER_LATENT:  # handle default case
            return self.get_single_autoencoder_context()
        else:
            return self.autoencoder_context_by_node_type.get(node_type, None)

    @property
    def has_single_autoencoder_context(self) -> bool:
        return len(self.autoencoder_context_by_node_type) == 1

    def get_single_autoencoder_context(self) -> AutoencoderContext:
        assert self.has_single_autoencoder_context
        return next(iter(self.autoencoder_context_by_node_type.values()))

    def get_autoencoder(
        self, layer_index: LayerIndex, node_type: NodeType | None = None
    ) -> Autoencoder:
        autoencoder_context = self.get_autoencoder_context(node_type)
        assert autoencoder_context is not None
        return autoencoder_context.get_autoencoder(layer_index)

    def warmup(self) -> None:
        """Load all autoencoders into memory."""
        for node_type, autoencoder_context in self.autoencoder_context_by_node_type.items():
            print(f"Warming up autoencoder {node_type}")
            autoencoder_context.warmup()


def get_decoder_weight(autoencoder: Autoencoder) -> torch.Tensor:
    return autoencoder.decoder.weight.T  # shape (n_latents, d_ff)


def get_autoencoder_output_weight_by_layer_index(
    autoencoder_context: AutoencoderContext,
) -> dict[LayerIndex, torch.Tensor]:
    return {
        layer_index: get_decoder_weight(
            autoencoder_context.get_autoencoder(layer_index)
        )  # shape (n_latents, d_ff)
        for layer_index in autoencoder_context.layer_indices
    }


ACTIVATION_FREQUENCY_THRESHOLD_FOR_DEAD_LATENTS = 1e-8


def count_dead_latents(autoencoder: Autoencoder) -> int:
    if hasattr(autoencoder, "latents_activation_frequency"):
        if torch.all(autoencoder.latents_activation_frequency == 0):
            raise ValueError("latents_activation_frequency is all zeros, all latents are dead.")
        dead_latents_mask = (
            autoencoder.latents_activation_frequency
            < ACTIVATION_FREQUENCY_THRESHOLD_FOR_DEAD_LATENTS
        )
        num_dead_latents = int(dead_latents_mask.sum().item())
        return num_dead_latents
    else:
        return 0


def omit_least_active_latents(
    autoencoder: Autoencoder,
    n_latents_to_keep: int,
    # if preserve_indices=True, ignore the stored activation frequencies, and keep the first indices.
    # this is to preserve latent indices compared to the original autoencoder.
    preserve_indices: bool = True,
) -> Autoencoder:
    n_latents_original = int(autoencoder.latent_bias.shape[0])
    if n_latents_to_keep >= n_latents_original:
        return autoencoder
    device: torch.device = autoencoder.encoder.weight.device

    # create the dead latent mask (True for live latents, False for dead latents)
    mask = torch.ones(n_latents_original, dtype=torch.bool, device=device)
    if preserve_indices or not hasattr(autoencoder, "latents_activation_frequency"):
        # drop the last latents
        mask[n_latents_to_keep:] = 0
    else:
        # drop the least active latents
        order = torch.argsort(autoencoder.latents_activation_frequency, descending=True)
        mask[order[n_latents_to_keep:]] = 0

    # apply the mask to a new autoencoder
    n_latents = int(mask.sum().item())
    d_model = autoencoder.pre_bias.shape[0]
    new_autoencoder = Autoencoder(n_latents, d_model).to(device)
    new_autoencoder.encoder.weight.data = autoencoder.encoder.weight[mask, :].clone()
    new_autoencoder.decoder.weight.data = autoencoder.decoder.weight[:, mask].clone()
    new_autoencoder.latent_bias.data = autoencoder.latent_bias[mask].clone()
    new_autoencoder.stats_last_nonzero.data = autoencoder.stats_last_nonzero[mask].clone()
    if hasattr(autoencoder, "latents_mean_square"):
        new_autoencoder.register_buffer(
            "latents_mean_square", torch.zeros(n_latents, dtype=torch.float)
        )
        new_autoencoder.latents_mean_square.data = autoencoder.latents_mean_square[mask].clone()
    if hasattr(autoencoder, "latents_activation_frequency"):
        new_autoencoder.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        new_autoencoder.latents_activation_frequency.data = (
            autoencoder.latents_activation_frequency[mask].clone()
        )
    return new_autoencoder
