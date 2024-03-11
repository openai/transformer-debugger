"""This file contains the primary code for the ScalarDeriver class."""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch

from neuron_explainer.activations.derived_scalars.activations_and_metadata import (
    ActivationsAndMetadata,
    RawActivationStore,
)
from neuron_explainer.activations.derived_scalars.config import DstConfig
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.locations import (
    DEFAULT_LAYER_INDEXER,
    LayerIndexer,
    NoLayersLayerIndexer,
    StaticLayerIndexer,
    get_location_within_layer_for_dst,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    ActivationLocationTypeAndPassType,
    Dimension,
    LayerIndex,
    LocationWithinLayer,
    PassType,
)

### SHARED CODE FOR DERIVING SCALARS FROM ACTIVATIONS ###


@dataclass(frozen=True)
class DerivedScalarTypeAndPassType:
    dst: DerivedScalarType
    pass_type: PassType


class ScalarSource(ABC):
    pass_type: PassType
    layer_indexer: LayerIndexer

    @property
    @abstractmethod
    def exists_by_default(self) -> bool:
        # returns True if the activation is instantiated by default in a normal transformer forward pass
        # this is False for activations related to autoencoders or for non-trivial derived scalars
        pass

    @property
    @abstractmethod
    def dst(self) -> DerivedScalarType:
        pass

    @property
    def dst_and_pass_type(self) -> "DerivedScalarTypeAndPassType":
        return DerivedScalarTypeAndPassType(
            self.dst,
            self.pass_type,
        )

    @property
    @abstractmethod
    def sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        pass

    @property
    @abstractmethod
    def location_within_layer(self) -> LocationWithinLayer | None:
        pass

    @property
    def layer_index(self) -> LayerIndex:
        """Convenience method to get the single layer index associated with this ScalarSource, if such a single layer index
        exists. Throws an error if it does not."""
        assert isinstance(self.layer_indexer, StaticLayerIndexer), (
            self.layer_indexer,
            "ScalarSource.layer_index should only be called for ScalarSource StaticLayerIndexer",
        )
        return self.layer_indexer.layer_index

    @abstractmethod
    def derive_from_raw(
        self,
        raw_activation_store: RawActivationStore,
        desired_layer_indices: (
            list[LayerIndex] | None
        ),  # indicates layer indices to keep; None indicates keep all
    ) -> ActivationsAndMetadata:
        """Given raw activations, derive the scalar value. desired_layer_indices is a list of layer indices to include in the output; None indicates all layers,
        while [None] indicates activations not indexed by layers (e.g. from the embedding)."""
        pass


# note that this class, inheriting from ActivationLocationTypeAndPassType, becomes a
# base class. This needs to be a separate object from ActivationLocationTypeAndPassType,
# and located within this file, because ScalarSource needs to know about DerivedScalarTypes,
# which are defined within the derived_scalars/ directory
class RawScalarSource(ActivationLocationTypeAndPassType, ScalarSource):
    def __init__(
        self,
        activation_location_type: ActivationLocationType,
        pass_type: PassType,
        layer_indexer: LayerIndexer = DEFAULT_LAYER_INDEXER,
    ) -> None:
        super().__init__(activation_location_type, pass_type)
        self.layer_indexer = layer_indexer
        if activation_location_type.has_no_layers:
            assert isinstance(layer_indexer, NoLayersLayerIndexer), self

    @property
    def dst(self) -> DerivedScalarType:
        return DerivedScalarType.from_activation_location_type(self.activation_location_type)

    @property
    def sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        return (self.activation_location_type_and_pass_type,)

    @property
    def exists_by_default(self) -> bool:
        return self.activation_location_type.exists_by_default

    @property
    def location_within_layer(self) -> LocationWithinLayer | None:
        return self.activation_location_type.location_within_layer

    @property
    def activation_location_type_and_pass_type(self) -> ActivationLocationTypeAndPassType:
        return ActivationLocationTypeAndPassType(self.activation_location_type, self.pass_type)

    def derive_from_raw(
        self,
        raw_activation_store: RawActivationStore,
        desired_layer_indices: (
            list[LayerIndex] | None
        ),  # indicates layer indices to keep; None indicates keep all
    ) -> ActivationsAndMetadata:
        return raw_activation_store.get_activations_and_metadata(
            self.activation_location_type,
            self.pass_type,
        ).apply_layer_indexer(self.layer_indexer, desired_layer_indices)


class DerivedScalarSource(ScalarSource):
    scalar_deriver: "ScalarDeriver"

    def __init__(
        self,
        scalar_deriver: "ScalarDeriver",
        pass_type: PassType,
        layer_indexer: LayerIndexer = DEFAULT_LAYER_INDEXER,
    ) -> None:
        self.scalar_deriver = scalar_deriver
        self.pass_type = pass_type
        self.layer_indexer = layer_indexer

    @property
    def exists_by_default(self) -> bool:
        return False

    @property
    def dst(self) -> DerivedScalarType:
        return self.scalar_deriver.dst

    @property
    def sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        return self.scalar_deriver.get_sub_activation_location_type_and_pass_types()

    @property
    def location_within_layer(self) -> LocationWithinLayer | None:
        return self.scalar_deriver.location_within_layer

    def derive_from_raw(
        self,
        raw_activation_store: RawActivationStore,
        desired_layer_indices: (
            list[LayerIndex] | None
        ),  # indicates layer indices to keep; None indicates keep all
    ) -> ActivationsAndMetadata:
        return self.scalar_deriver.derive_from_raw(
            raw_activation_store, self.pass_type
        ).apply_layer_indexer(self.layer_indexer, desired_layer_indices=desired_layer_indices)


@dataclass(frozen=True)
class ScalarDeriver:
    """Contains the information necessary for specifying some function of one or more activations,
    (this function can be as simple as the identity function). This includes: what activations are required
    to compute it; a function that takes in ActivationsAndMetadata for each of those activations and
    returns a ActivationsAndMetadata for the derived scalar; and a function that
    returns the shape you expect the derived scalar to have for each token (e.g. one float per attention head,
    one float per layer, etc.).
    The function for computing this derived scalar on the forward pass can be different from the function for computing
    its gradient on the backward pass, so the pass type must also be an argument to the function that computes the scalar.
    A HookLocationType describes the type of activation that is saved during inference, and a ScalarDeriver describes the
    type of "derived" scalar computed from those activations after they are read from disk. In the simplest case, a
    derived scalar can be computed directly from the saved activations with an identity transformation (e.g. a single MLP
    activation is saved during inference, and the derived scalar is the same MLP activation)."""

    dst: DerivedScalarType

    """
    Dataclass with fields needed to construct a ScalarDeriver for this DerivedScalarType; e.g. derived scalars
    computed using model weights will require at minimum the model_name to load the weights.
    """
    dst_config: DstConfig

    """
    Contains ActivationLocationTypes or other ScalarDerivers, and corresponding pass directions (forward or backward) that are
    required to compute this derived scalar type. These are loaded from disk and passed to the
    tensor_calculate_derived_scalar_fn as a single tuple argument.
    """
    sub_scalar_sources: tuple[ScalarSource, ...]

    """
    A function that takes a tuple of tensors, a layer index, and a pass type, and returns
    a tensor containing the derived scalar values. layer_index can be None in case of activation
    location types that don't have layer indices, like embeddings.
    """
    tensor_calculate_derived_scalar_fn: Callable[
        [tuple[torch.Tensor, ...], LayerIndex, PassType], torch.Tensor
    ]

    """In cases where a ScalarDeriver is a transform applied to another scalar deriver, the location within a layer associated
    with the resulting scalar deriver is taken to be the same as the location within a layer associated with the original scalar deriver.
    See definition of LocationWithinLayer in model_component_registry.py for more details."""
    _specified_location_within_layer: LocationWithinLayer | None = None

    @property
    def device_for_raw_activations(self) -> torch.device:
        """Which device to read raw activations onto."""
        return self.dst_config.get_device()

    @property
    def shape_of_activation_per_token_spec(self) -> tuple[Dimension, ...]:
        # first dimension is num_sequence_tokens; this can be either the literal number of tokens in a sequence or
        # the number of token pairs in a sequence
        return self.dst.shape_spec_per_token_sequence[1:]

    @property
    def location_within_layer(self) -> LocationWithinLayer | None:
        """An activation location type at a topologically equivalent point in the network, in terms of which
        residual stream locations precede and follow it."""
        specified_location_within_layer = self._specified_location_within_layer
        dst_location_within_layer = get_location_within_layer_for_dst(self.dst, self.dst_config)
        if specified_location_within_layer is not None and dst_location_within_layer is not None:
            assert specified_location_within_layer == dst_location_within_layer
        consensus_location_within_layer = (
            specified_location_within_layer or dst_location_within_layer
        )
        return consensus_location_within_layer

    def _check_dst_and_pass_types(
        self, activation_data_tuple: tuple[ActivationsAndMetadata, ...]
    ) -> None:
        """Check that the derived scalar types and pass types of the raw activations match
        the order of the dsts and pass types in self.get_sub_dst_and_pass_types().
        """
        assert len(activation_data_tuple) == len(self.get_sub_dst_and_pass_types()), (
            [activation_data.dst for activation_data in activation_data_tuple],
            [
                sub_dst_and_pass_type.dst
                for sub_dst_and_pass_type in self.get_sub_dst_and_pass_types()
            ],
        )
        for activation_data, sub_dst_and_pass_type in zip(
            activation_data_tuple, self.get_sub_dst_and_pass_types()
        ):
            assert (
                activation_data.dst == sub_dst_and_pass_type.dst
            ), f"{activation_data.dst=}, {sub_dst_and_pass_type.dst=}"
            assert activation_data.pass_type == sub_dst_and_pass_type.pass_type, (
                f"{self.dst=}, "
                f"{activation_data.dst=}, "
                f"{activation_data.pass_type=}, {sub_dst_and_pass_type.pass_type=}"
            )
            assert activation_data.pass_type == sub_dst_and_pass_type.pass_type
        return

    def activations_and_metadata_calculate_derived_scalar_fn(
        self, activation_data_tuple: tuple[ActivationsAndMetadata, ...], pass_type: PassType
    ) -> ActivationsAndMetadata:
        self._check_dst_and_pass_types(activation_data_tuple)
        for activation_data in activation_data_tuple:
            assert len(activation_data.activations_by_layer_index) > 0, (
                f"{activation_data.activations_by_layer_index=}"
                f"{activation_data.dst=}"
                f"{activation_data.pass_type=}"
            )
        activation_data = activation_data_tuple[0]
        filtered_activation_data = activation_data.filter_layers(
            layer_indices=self.dst_config.layer_indices
        )
        if len(activation_data_tuple) == 1:

            def _calculate_derived_scalar_fn(
                activations: torch.Tensor,
                layer_index: LayerIndex,
            ) -> torch.Tensor:
                return self.tensor_calculate_derived_scalar_fn(
                    (activations,), layer_index, pass_type
                )

            return filtered_activation_data.apply_layerwise_transform_fn_to_activations(
                layerwise_transform_fn=_calculate_derived_scalar_fn,
                output_dst=self.dst,
                output_pass_type=pass_type,
            )
        elif len(activation_data_tuple) >= 2:

            def _calculate_multi_arg_derived_scalar_fn(
                *args: torch.Tensor,
                layer_index: LayerIndex,
            ) -> torch.Tensor:
                return self.tensor_calculate_derived_scalar_fn(tuple(args), layer_index, pass_type)

            other_filtered_activation_data_tuple = tuple(
                activation_data.filter_layers(layer_indices=self.dst_config.layer_indices)
                for activation_data in activation_data_tuple[1:]
            )

            return filtered_activation_data.apply_layerwise_transform_fn_to_multiple_activations(
                # care should be taken in a dictionary comprehension of callables that the
                # variables (i.e. layer_index) are bound at time of creation, not at time of execution
                # partial accomplishes this
                layerwise_transform_fn=_calculate_multi_arg_derived_scalar_fn,
                others=other_filtered_activation_data_tuple,
                output_dst=self.dst,
                output_pass_type=pass_type,
            )
        else:
            raise NotImplementedError(
                f"ScalarDeriver.activations_and_metadata_calculate_derived_scalar_fn not implemented for "
                f"{len(activation_data_tuple)=}"
            )

    def derive_from_raw(
        self,
        raw_activation_store: RawActivationStore,
        pass_type: PassType,
    ) -> ActivationsAndMetadata:
        desired_layer_indices = None
        sub_activations_list = []
        for sub_scalar_source in self.get_sub_scalar_sources():
            sub_activation_data = sub_scalar_source.derive_from_raw(
                raw_activation_store, desired_layer_indices=desired_layer_indices
            )
            sub_activations_list.append(sub_activation_data)
            if len(sub_activations_list) == 1:
                desired_layer_indices = list(sub_activations_list[0].layer_indices)
        return self.activations_and_metadata_calculate_derived_scalar_fn(
            tuple(sub_activations_list), pass_type
        )

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "dst": self.dst,
            "dst_config": self.dst_config,
        }

    def get_sub_dst_and_pass_types(self) -> tuple[DerivedScalarTypeAndPassType, ...]:
        return tuple(
            sub_scalar_source.dst_and_pass_type for sub_scalar_source in self.sub_scalar_sources
        )

    def get_sub_scalar_sources(self) -> tuple[ScalarSource, ...]:
        return self.sub_scalar_sources

    def get_sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        sub_activation_location_type_and_pass_types_list = []
        for scalar_source in self.get_sub_scalar_sources():
            sub_activation_location_type_and_pass_types_list.extend(
                list(scalar_source.sub_activation_location_type_and_pass_types)
            )
        return tuple(sub_activation_location_type_and_pass_types_list)

    @property
    def n_input_tensors(self) -> int:
        # the number of arguments expected by the top-level function. Note that this is not necessarily
        # the same as the number of sub_activation_location_type_and_pass_types; some of these might be
        # consumed by lower-level ScalarDerivers, and combined into single tensors passed to the top-level
        # function.
        return len(self.get_sub_dst_and_pass_types())

    @property
    def n_total_required_tensors(self) -> int:
        # the number of tensors required to compute the derived scalar, including those that are
        # passed to lower-level ScalarDerivers
        return len(self.get_sub_activation_location_type_and_pass_types())

    @property
    def derivable_pass_types(self) -> tuple[PassType, ...]:
        # ScalarDerivers are configurable to support either only computing a
        # scalar on the forward pass, or computing it on both the forward and
        # the backward pass. Supporting the backward pass requires more kinds
        # of raw activations in general.
        if self.dst_config.derive_gradients:
            return (PassType.FORWARD, PassType.BACKWARD)
        else:
            return (PassType.FORWARD,)

    def apply_transform_fn_to_output(
        self,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
        pass_type_to_transform: PassType,
        output_dst: DerivedScalarType,
    ) -> "ScalarDeriver":
        """Converts one ScalarDeriver to another, by applying a tensor -> tensor function to the output.
        The tensor -> tensor function takes a tensor, a layer index, and a pass type, and returns a tensor,
        so that it can depend on layer and pass type."""

        def layerwise_transform_fn(
            tensor: torch.Tensor,
            layer_index: LayerIndex,
            pass_type: PassType,
        ) -> torch.Tensor:
            return transform_fn(tensor)

        return self.apply_layerwise_transform_fn_to_output(
            layerwise_transform_fn=layerwise_transform_fn,
            pass_type_to_transform=pass_type_to_transform,
            output_dst=output_dst,
        )

    def apply_layerwise_transform_fn_to_output(
        self,
        layerwise_transform_fn: Callable[[torch.Tensor, LayerIndex, PassType], torch.Tensor],
        pass_type_to_transform: PassType,
        output_dst: DerivedScalarType,
    ) -> "ScalarDeriver":
        """Converts one ScalarDeriver to another, by applying a tensor -> tensor function to the output.
        The tensor -> tensor function takes a tensor, a layer index, and a pass type, and returns a tensor,
        so that it can depend on layer and pass type."""
        sub_scalar_sources = (DerivedScalarSource(self, pass_type=pass_type_to_transform),)

        def tensor_calculate_derived_scalar_fn(
            activation_data_tuple: tuple[torch.Tensor, ...],
            layer_index: LayerIndex,
            pass_type: PassType,
        ) -> torch.Tensor:
            assert len(activation_data_tuple) == 1
            return layerwise_transform_fn(activation_data_tuple[0], layer_index, pass_type)

        return dataclasses.replace(
            self,
            dst=output_dst,
            sub_scalar_sources=sub_scalar_sources,
            tensor_calculate_derived_scalar_fn=tensor_calculate_derived_scalar_fn,
            _specified_location_within_layer=self.location_within_layer,
        )

    def apply_layerwise_transform_fn_to_output_and_other_tensor(
        self,
        layerwise_transform_fn: Callable[..., torch.Tensor],
        pass_type_to_transform: PassType,
        output_dst: DerivedScalarType,
        other_scalar_source: ScalarSource,
    ) -> "ScalarDeriver":
        """Converts one ScalarDeriver to another, by applying a two tensor -> tensor function to the output + an additional activation tensor.
        The tensor -> tensor function takes two tensors, a layer index, and a pass type, and returns a tensor,
        so that it can depend on layer and pass type."""
        sub_scalar_sources = (
            DerivedScalarSource(
                self, pass_type=pass_type_to_transform, layer_indexer=DEFAULT_LAYER_INDEXER
            ),
            other_scalar_source,
        )

        def tensor_calculate_derived_scalar_fn(
            activation_data_tuple: tuple[torch.Tensor, ...],
            layer_index: LayerIndex,
            pass_type: PassType,
        ) -> torch.Tensor:
            assert len(activation_data_tuple) == 2, [t.shape for t in activation_data_tuple]
            return layerwise_transform_fn(*activation_data_tuple, layer_index, pass_type)

        return dataclasses.replace(
            self,
            dst=output_dst,
            sub_scalar_sources=sub_scalar_sources,
            tensor_calculate_derived_scalar_fn=tensor_calculate_derived_scalar_fn,
            _specified_location_within_layer=self.location_within_layer,
        )
