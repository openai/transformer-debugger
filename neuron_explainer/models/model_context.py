import os
from abc import ABC, abstractmethod
from typing import Any

import blobfile as bf
import tiktoken
import torch
import torch.nn as nn

from neuron_explainer.models import Transformer, TransformerConfig
from neuron_explainer.models.inference_engine_type_registry import InferenceEngineType
from neuron_explainer.models.model_component_registry import (
    Dimension,
    LayerIndex,
    WeightLocationType,
    get_dimension_index_of_weight_location_type,
    weight_shape_by_location_type,
)
from neuron_explainer.models.model_registry import get_standard_model_spec

ALLOWED_SPECIAL_TOKENS = {"<|endoftext|>"}


class InvalidTokenException(Exception):
    pass


class ModelContext(ABC):
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.model_name = model_name
        self.device = device

    # takes a WeightLocationType and optional layer
    # returns the tensor
    @abstractmethod
    def _get_weight_helper(
        self,
        location_type: WeightLocationType,
        layer: LayerIndex = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ...

    def get_weight(
        self,
        location_type: WeightLocationType,
        layer: LayerIndex = None,
        normalize_dim: Dimension | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Returns the specified weights, with shape checking and optional normalization.

        Tensors returned by this method are not cloned, so please be sure not to perform in-place
        edits on them!
        """
        assert (
            location_type in weight_shape_by_location_type
        ), f"location_type_str {location_type} not found"
        weight = self._get_weight_helper(
            location_type=location_type, layer=layer, device=device or self.device
        )

        if normalize_dim is not None:
            weight = nn.functional.normalize(
                weight,
                dim=get_dimension_index_of_weight_location_type(location_type, normalize_dim),
            )
        weight_shape_spec = weight_shape_by_location_type[location_type]
        expected_shape = self.get_shape_from_spec(weight_shape_spec)
        assert (
            weight.shape == expected_shape
        ), f"Expected shape {expected_shape} but got {weight.shape}"
        # We don't want to return tensors that have gradients enabled, so we detach. Ideally we'd
        # also clone since we don't want callers to inadvertently edit the weights, but doing so
        # uses a lot of memory, so instead we just ask politely in the docstring.
        return weight.detach()

    # get Encoding -> call this in the base class

    @abstractmethod
    def get_encoding(self) -> tiktoken.Encoding:
        ...

    def encode(self, string: str) -> list[int]:
        return self.get_encoding().encode(string, allowed_special=ALLOWED_SPECIAL_TOKENS)

    def decode_token(self, token: int) -> str:
        return self.get_encoding().decode([token])

    def decode(self, token_list: list[int]) -> str:
        return self.get_encoding().decode(token_list)

    def encode_token_str(self, token_str: str) -> int:
        token_int_list = self.encode(token_str)
        if len(token_int_list) != 1:
            raise InvalidTokenException(
                f"'{token_str}' decoded to {token_int_list}; wanted exactly 1 token"
            )
        return token_int_list[0]

    @abstractmethod
    def get_dim_size(self, model_dimension_spec: Dimension) -> int:
        ...

    def get_shape_from_spec(self, shape_spec: tuple[Dimension, ...]) -> tuple[int, ...]:
        expected_shape: tuple[int, ...] = tuple(
            self.get_dim_size(dimension_spec) if dimension_spec != Dimension.SINGLETON else 1
            for dimension_spec in shape_spec
        )

        return expected_shape

    @abstractmethod
    def get_or_create_model(self) -> Transformer:
        """Returns an instantiated model which can be used to run forward passes.

        The first call to this method results in a new model being created. Subsequent calls return
        the same cached model instance.
        """
        ...

    def decode_token_list(self, token_list: list[int]) -> list[str]:
        return [self.decode_token(token=token) for token in token_list]

    def encode_token_str_list(self, token_str_list: list[str]) -> list[int]:
        return [self.encode_token_str(token_str=token_str) for token_str in token_str_list]

    @classmethod
    def from_model_type(
        cls,
        model_type: str,
        inference_engine_type: InferenceEngineType = InferenceEngineType.STANDARD,
        **kwargs: Any,
    ) -> "ModelContext":
        device = kwargs.pop("device", get_default_device())
        if inference_engine_type == InferenceEngineType.STANDARD:
            return StandardModelContext(model_name=model_type, device=device, **kwargs)
        else:
            raise ValueError(f"Unsupported inference_engine_type {inference_engine_type}")

    @property
    def n_neurons(self) -> int:
        return self.get_dim_size(Dimension.MLP_ACTS)

    @property
    def n_attention_heads(self) -> int:
        return self.get_dim_size(Dimension.ATTN_HEADS)

    @property
    def n_layers(self) -> int:
        return self.get_dim_size(Dimension.LAYERS)

    @property
    def n_residual_stream_channels(self) -> int:
        return self.get_dim_size(Dimension.RESIDUAL_STREAM_CHANNELS)

    @property
    def n_vocab(self) -> int:
        return self.get_dim_size(Dimension.VOCAB_SIZE)

    @property
    def n_context(self) -> int:
        return self.get_dim_size(Dimension.MAX_CONTEXT_LENGTH)

    @abstractmethod
    def get_model_config_as_dict(self) -> dict[str, Any]:
        ...


# Note: If you're seeing mysterious crashes while running on a MacBook, try switching from "mps" to
# "cpu".
def get_default_device() -> torch.device:
    # TODO: Figure out why test_interactive_model.py crashes on the "mps" backend, then remove
    # this workaround.
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    elif torch.backends.mps.is_available() and not is_pytest:
        return torch.device("mps", 0)
    else:
        return torch.device("cpu")


class StandardModelContext(ModelContext):
    def __init__(self, model_name: str, device: torch.device | None = None) -> None:
        if device is None:
            device = get_default_device()
        super().__init__(model_name=model_name, device=device)
        self._model_spec = get_standard_model_spec(self.model_name)
        self.load_path = self._model_spec.model_path
        self._config = TransformerConfig.load(f"{self.load_path}/config.json")
        # Once a transformer has been created via get_or_create_model, we cache it. Subsequent calls
        # to get_or_create_model return the cached instance.
        self._cached_transformer: Transformer | None = None

    @classmethod
    def from_model_type(
        cls,
        model_type: str,
        inference_engine_type: InferenceEngineType = InferenceEngineType.STANDARD,
        **kwargs: Any,
    ) -> ModelContext:  # specifically a StandardModelContext, but to satisfy mypy
        assert (
            inference_engine_type == InferenceEngineType.STANDARD
        ), "don't set a different inference_engine_type kwarg here"
        model_context = super().from_model_type(
            model_type=model_type, inference_engine_type=InferenceEngineType.STANDARD, **kwargs
        )
        assert isinstance(model_context, StandardModelContext)
        return model_context

    def get_dim_size(self, model_dimension_spec: Dimension) -> int:
        # TODO(sbills): This should really be a match statement.
        dimension_by_dimension_spec: dict[Dimension, int] = {
            Dimension.MAX_CONTEXT_LENGTH: self._config.ctx_window,
            Dimension.RESIDUAL_STREAM_CHANNELS: self._config.d_model,
            Dimension.VOCAB_SIZE: self.get_encoding().n_vocab,
            Dimension.ATTN_HEADS: self._config.n_heads,
            Dimension.QUERY_AND_KEY_CHANNELS: self._config.d_head_qk,
            Dimension.VALUE_CHANNELS: self._config.d_head_v,
            Dimension.MLP_ACTS: self._config.d_ff,
            Dimension.MLP_ACTS: self._config.d_ff,
            Dimension.LAYERS: self._config.n_layers,
        }
        return dimension_by_dimension_spec[model_dimension_spec]

    def _get_weight_helper(
        self,
        location_type: WeightLocationType,
        layer: LayerIndex = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        info_by_type: dict[WeightLocationType, dict] = {
            WeightLocationType.MLP_TO_HIDDEN: dict(
                part=f"xf_layers.{layer}.mlp.in_layer.weight",
                reshape="hr->rh",
            ),
            WeightLocationType.MLP_TO_RESIDUAL: dict(
                part=f"xf_layers.{layer}.mlp.out_layer.weight",
                reshape="rh->hr",
            ),
            WeightLocationType.EMBEDDING: dict(
                part="tok_embed.weight",
            ),
            WeightLocationType.UNEMBEDDING: dict(
                part="unembed.weight",
                reshape="vr->rv",
            ),
            WeightLocationType.POSITION_EMBEDDING: dict(
                part="pos_embed.weight",
            ),
            WeightLocationType.ATTN_TO_QUERY: dict(
                part=f"xf_layers.{layer}.attn.q_proj.weight",
                split=(0, self._config.n_heads),
                reshape="hqr->hrq",
            ),
            WeightLocationType.ATTN_TO_KEY: dict(
                part=f"xf_layers.{layer}.attn.k_proj.weight",
                split=(0, self._config.n_heads),
                reshape="hkr->hrk",
            ),
            WeightLocationType.ATTN_TO_VALUE: dict(
                part=f"xf_layers.{layer}.attn.v_proj.weight",
                split=(0, self._config.n_heads),
                reshape="hvr->hrv",
            ),
            WeightLocationType.ATTN_TO_RESIDUAL: dict(
                part=f"xf_layers.{layer}.attn.out_proj.weight",
                split=(1, self._config.n_heads),
                reshape="rhv->hvr",
            ),
            WeightLocationType.LAYER_NORM_GAIN_FINAL: dict(
                part="final_ln.weight",
                broadcast=True,
            ),
            WeightLocationType.LAYER_NORM_BIAS_FINAL: dict(
                part="final_ln.bias",
            ),
            WeightLocationType.LAYER_NORM_GAIN_PRE_ATTN: dict(
                part=f"xf_layers.{layer}.ln_1.weight",
            ),
            WeightLocationType.LAYER_NORM_GAIN_PRE_MLP: dict(
                part=f"xf_layers.{layer}.ln_2.weight",
            ),
        }
        info = info_by_type.get(location_type)
        if info is None:
            raise NotImplementedError(f"Unsupported weight location type: {location_type}")
        part = info["part"]
        split = info.get("split")
        reshape = info.get("reshape")
        if self._cached_transformer is None:
            with bf.BlobFile(f"{self.load_path}/model_pieces/{part}.pt", "rb") as f:
                weight = torch.load(f, map_location=device or self.device)
        else:
            weight = self._cached_transformer.state_dict()[part].to(device or self.device)

        if split is not None:
            (dim_split, n_split) = split
            w_shape = list(weight.shape)
            w_shape_new = (
                w_shape[:dim_split]
                + [n_split, w_shape[dim_split] // n_split]
                + w_shape[dim_split + 1 :]
            )
            weight = weight.reshape(*w_shape_new)
        if reshape is not None:
            weight = torch.einsum(reshape, weight)
        # Some tensors are sometimes stored with a subset of dimensions and then broadcasted in the model
        # E.g. the final layer norm gain is stored as a scalar
        # Broadcast flag indicates that we should broadcast them to the expected shape
        broadcast = info.get("broadcast")
        if broadcast is True:
            expected_shape = self.get_shape_from_spec(weight_shape_by_location_type[location_type])
            weight = weight.expand(expected_shape)
        return weight

    def get_or_create_model(
        self,
        device: torch.device | None = None,
        simplify: bool = False,
    ) -> Transformer:
        if self._cached_transformer is None:
            self._cached_transformer = Transformer.load(
                self.load_path, simplify=simplify, device=device or self.device
            )

        return self._cached_transformer

    def get_encoding(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding(self._config.enc)

    def get_model_config_as_dict(self) -> dict[str, Any]:
        return self._config.to_dict()


class StubModelContext(ModelContext):
    # TODO: maybe make a unified interface for the Config objects of ModelContext objects, and
    # have this be a StubConfig instead of a StubContext
    """This is a fake model context object for use in testing. It just works as a holder for
    a mapping from model dimension to size (int)."""

    def __init__(
        self,
        size_by_model_dimension_spec: dict[Dimension, int],
    ):
        super().__init__(model_name="stub", device=torch.device("cpu"))
        self._size_by_model_dimension_spec = size_by_model_dimension_spec

    def _get_weight_helper(
        self,
        location_type: WeightLocationType,
        layer: LayerIndex = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_encoding(self) -> tiktoken.Encoding:
        raise NotImplementedError

    def get_or_create_model(self) -> Transformer:
        raise NotImplementedError

    def get_model_config_as_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_dim_size(self, model_dimension_spec: Dimension) -> int:
        if model_dimension_spec in self._size_by_model_dimension_spec:
            return self._size_by_model_dimension_spec[model_dimension_spec]
        else:
            raise NotImplementedError


# TODO: make this robust to whether the transformer is 'simplified' in our terminology
# once the .simplify() operation is extended to cover final layer norm gain
def get_unembedding_with_ln_gain(model_context: ModelContext) -> torch.Tensor:
    """
    returns an unembedding matrix multiplied by the layer norm gain (a d_model-dimensional vector)
    for the final layer
    """
    Unemb_without_ln_gain = model_context.get_weight(
        location_type=WeightLocationType.UNEMBEDDING,
        device=model_context.device,
    )
    ln_gain_final = model_context.get_weight(
        location_type=WeightLocationType.LAYER_NORM_GAIN_FINAL,
        device=model_context.device,
    )
    return torch.einsum("ov,o->ov", Unemb_without_ln_gain, ln_gain_final)


def get_embedding(model_context: ModelContext) -> torch.Tensor:
    """
    returns an embedding matrix. Note that there is no layer norm in between the embedding tensor and
    the residual stream
    """
    return model_context.get_weight(
        location_type=WeightLocationType.EMBEDDING,
        device=model_context.device,
    )
