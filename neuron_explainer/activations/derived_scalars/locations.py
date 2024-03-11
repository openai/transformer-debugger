"""
This file contains code related to specifying the locations of derived scalars, and their inputs,
within the residual stream.
"""

from abc import ABC, abstractmethod
from typing import Literal, Sequence

from neuron_explainer.activations.derived_scalars.config import DstConfig
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import ActivationIndex
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    LayerIndex,
    LocationWithinLayer,
    NodeType,
    PassType,
)


class LayerIndexer(ABC):
    """A LayerIndexer is a function that maps from a list of indices in an original ActivationsAndMetadata object
    and a list of indices in a reindexed ActivationsAndMetadata object. It can do things like:
    - replace the activations at every layer with a reference to the activations at a single layer
    - replace the activations at every layer with a reference to a single activation from an ActivationLocationType
    that doesn't use layers (e.g. residual stream post embedding)
    - replace the activations at every layer with a reference to the activations one layer earlier
    - keep the activations at every layer the same
    DST computation typically acts on the activations at the same layer for each of several layers in an ActivationsAndMetadata object.
    When the computation requires activations from multiple distinct layers to compute the result for a given layer, this class
    handles the remapping so that downstream code can act on each layer independently."""

    @abstractmethod
    def __call__(self, layer_indices: list[LayerIndex]) -> Sequence[LayerIndex | Literal["Dummy"]]:
        # given a list of layer indices to an original ActivationsAndMetadata object, return a list of layer indices
        # with which to index the activations_by_layer_index of the original object in order to obtain the reindexed
        # activations_by_layer_index of the new object
        # int refers to a normal layer index
        # None refers to an activation with no layer index (e.g. embeddings)
        # "Dummy" is used when the reindexed
        # ActivationsAndMetadata object does not require the activation from the original object at that layer index,
        # for example if it's the input to a derived scalar computation that doesn't require every activation at every
        # layer index
        pass


class IdentityLayerIndexer(LayerIndexer):
    """Sometimes computing derived scalar D at layer L requires Scalar S from layer L, and Scalar T from layer L. In this case no changes are needed
    to the layer indices of the activations_by_layer_index in the ActivationsAndMetadata object. This is used for such cases (it does a no-op).
    """

    def __init__(self) -> None:
        pass

    def __call__(self, layer_indices: list[LayerIndex]) -> list[LayerIndex]:
        return layer_indices

    def __repr__(self) -> str:
        return "IdentityLayerIndexer()"


class OffsetLayerIndexer(LayerIndexer):
    """Sometimes computing derived scalar D at layer L requires Scalar S from layer L, and Scalar T from layer L-1.
    This is used for populating an ActivationsAndMetadata object at each layer index with references to the activations at the previous layer index.
    """

    def __init__(self, layer_index_offset: int) -> None:
        self.layer_index_offset = layer_index_offset

    def __call__(self, layer_indices: list[LayerIndex]) -> list[LayerIndex | Literal["Dummy"]]:
        def _dummy_if_invalid(
            layer_index: LayerIndex, valid_indices: set[LayerIndex]
        ) -> LayerIndex | Literal["Dummy"]:
            if layer_index in valid_indices:
                return layer_index
            else:
                # this value represents the fact that the layer index is not needed for this computation
                # callers are free to use a dummy tensor in place of the activations at this layer index,
                # knowing that downstream DST calculations are intended to be independent of the activation
                # tensor provided at this layer index
                return "Dummy"

        assert all(layer_index is not None for layer_index in layer_indices)
        # source_layer_indices satisfy:
        # target_layer_indices = layer_indices
        # for target_layer_index, source_layer_index in zip(target_layer_indices, source_layer_indices):
        #    target_activations_by_layer_index[target_layer_index] = source_activations_by_layer_index[source_layer_index] # (or a dummy tensor, if the index is "Dummy")
        source_layer_indices = [layer_index + self.layer_index_offset for layer_index in layer_indices]  # type: ignore
        # 'invalid' layer indices are those not in starting_layer_indices; starting_layer_indices mapped to unneeded layer indices are considered "Unneeded"
        return [
            _dummy_if_invalid(source_layer_index, set(layer_indices))
            for source_layer_index in source_layer_indices
        ]

    def __repr__(self) -> str:
        return f"OffsetLayerIndexer(layer_index_offset={self.layer_index_offset})"


class StaticLayerIndexer(LayerIndexer, ABC):
    """A subset of LayerIndexers have a single layer_index associated with them. This gives those LayerIndexers a
    common abstract interface."""

    layer_index: LayerIndex

    def __call__(self, layer_indices: list[LayerIndex]) -> list[LayerIndex]:
        # this says to use the same activation tensor at every layer index requested; each layer index
        # is mapped to the same constant (or None) layer index
        return [self.layer_index for _ in layer_indices]


class ConstantLayerIndexer(StaticLayerIndexer):
    """Sometimes computing derived scalar D at layer L requires Scalar S from layer L, and Scalar T from layer C (independent of L).
    This is used for populating an ActivationsAndMetadata object with references to the same activation tensor (from layer C) at every layer index L.
    """

    def __init__(self, constant_layer_index: int) -> None:
        self.layer_index = constant_layer_index

    def __repr__(self) -> str:
        return f"ConstantLayerIndexer(constant_layer_index={self.layer_index})"


class NoLayersLayerIndexer(StaticLayerIndexer):
    """Sometimes computing derived scalar D at layer L requires Scalar S from layer L, and Scalar T which doesn't have layers.
    This is used for populating an ActivationsAndMetadata object with references to the same activation tensor (from a location type with no layers, i.e.
    at the index None) at every layer index L."""

    def __init__(self) -> None:
        self.layer_index = None

    def __repr__(self) -> str:
        return "NoLayersLayerIndexer()"


DEFAULT_LAYER_INDEXER = IdentityLayerIndexer()


def precedes_final_layer(
    derived_scalar_location_within_layer: LocationWithinLayer | None,
    derived_scalar_layer_index: LayerIndex,
    final_residual_location_within_layer: LocationWithinLayer | None,
    final_residual_layer_index: LayerIndex,
) -> bool:
    """Returns True if the derived scalar at a given layer_index precedes the final residual stream derived scalar
    at a specified layer_index"""
    # return True if the derived scalar at a given layer_index precedes the final residual stream layer_index
    if derived_scalar_layer_index is None:
        return True  # activations with no layer_index are assumed to precede
        # all activations with layer_index; note that according to current conventions
        # this is true for all residual stream activations (not true e.g. for token logits)
    elif final_residual_layer_index is None:
        assert derived_scalar_layer_index is not None
        return False  # activations with layer_index precede activations with no layer_index
    elif derived_scalar_layer_index < final_residual_layer_index:
        return True
    elif derived_scalar_layer_index == final_residual_layer_index:
        if derived_scalar_location_within_layer is None:
            raise ValueError(
                "derived_scalar_location_within_layer must be provided in case of equal layer indices"
            )
        if final_residual_location_within_layer is None:
            raise ValueError(
                "final_residual_location_within_layer must be provided in case of equal layer indices"
            )
        if derived_scalar_location_within_layer < final_residual_location_within_layer:
            # location_within_layer inherits from int; therefore they are straightforwardly comparable
            return True
        else:
            return False
    else:
        assert derived_scalar_layer_index > final_residual_layer_index
        return False


def get_location_within_layer_for_dst(
    dst: DerivedScalarType,
    dst_config: DstConfig,
) -> LocationWithinLayer | None:
    """Determines the location within a layer for DSTs which are not associated with an activation
    location type, or whose location within a layer depends on information in the DstConfig (e.g.
    autoencoder related DSTs). Defining new direct write related DSTs may require additional entries
    here."""
    if dst.location_within_layer is not None:
        # this might be determinable from the DST alone, in which case return it right away
        return dst.location_within_layer
    else:
        match dst.node_type:
            case (
                NodeType.AUTOENCODER_LATENT
                | NodeType.MLP_AUTOENCODER_LATENT
                | NodeType.ATTENTION_AUTOENCODER_LATENT
            ):
                autoencoder_context = dst_config.get_autoencoder_context(dst.node_type)
                if autoencoder_context is not None:
                    return autoencoder_context.dst.location_within_layer
                else:
                    return None
            case NodeType.RESIDUAL_STREAM_CHANNEL:
                match dst:
                    case DerivedScalarType.ATTN_WRITE:
                        return LocationWithinLayer.ATTN
                    case DerivedScalarType.PREVIOUS_LAYER_RESID_POST_MLP:
                        return LocationWithinLayer.END_OF_PREV_LAYER
                    case _:
                        return None
            case _:
                return None


def get_previous_residual_dst_for_node_type(
    node_type: NodeType,
    autoencoder_dst: DerivedScalarType | None,
) -> DerivedScalarType:
    """This function returns the DerivedScalarType of the residual stream that precedes the node
    type specified. autoencoder_context is only required if node_type is NodeType.ONLINE_AUTOENCODER.
    """
    match node_type:
        case NodeType.ATTENTION_HEAD:
            return DerivedScalarType.PREVIOUS_LAYER_RESID_POST_MLP
        case NodeType.MLP_NEURON:
            return DerivedScalarType.RESID_POST_ATTN
        case (
            NodeType.AUTOENCODER_LATENT
            | NodeType.MLP_AUTOENCODER_LATENT
            | NodeType.ATTENTION_AUTOENCODER_LATENT
        ):
            assert autoencoder_dst is not None, node_type
            match autoencoder_dst.node_type:
                case NodeType.RESIDUAL_STREAM_CHANNEL:
                    match autoencoder_dst:
                        case DerivedScalarType.RESID_DELTA_ATTN:
                            return get_previous_residual_dst_for_node_type(
                                node_type=NodeType.ATTENTION_HEAD,
                                autoencoder_dst=None,
                            )
                        case DerivedScalarType.RESID_DELTA_MLP:
                            return get_previous_residual_dst_for_node_type(
                                node_type=NodeType.MLP_NEURON,
                                autoencoder_dst=None,
                            )
                        case _:
                            raise NotImplementedError(autoencoder_dst)
                case _:
                    return get_previous_residual_dst_for_node_type(
                        node_type=autoencoder_dst.node_type,
                        autoencoder_dst=None,
                    )
        case _:
            raise NotImplementedError(node_type)


def get_activation_index_for_residual_dst(
    dst: DerivedScalarType,
    layer_index: int,
) -> ActivationIndex:
    """
    This returns an ActivationIndex corresponding to a residual stream activation location
    at a given layer_index; handles the indexing logic in the case of PREVIOUS_LAYER_RESID_POST_MLP.
    The ActivationIndex returned corresponds to the entire residual stream activation tensor for the
    layer.
    """
    assert dst.node_type == NodeType.RESIDUAL_STREAM_CHANNEL
    match dst:
        case DerivedScalarType.PREVIOUS_LAYER_RESID_POST_MLP:
            if layer_index == 0:
                return ActivationIndex(
                    activation_location_type=ActivationLocationType.RESID_POST_EMBEDDING,
                    layer_index=None,
                    tensor_indices=(),
                    pass_type=PassType.FORWARD,
                )
            else:
                return ActivationIndex(
                    activation_location_type=ActivationLocationType.RESID_POST_MLP,
                    layer_index=layer_index - 1,
                    tensor_indices=(),
                    pass_type=PassType.FORWARD,
                )
        case DerivedScalarType.RESID_POST_MLP:
            return ActivationIndex(
                activation_location_type=ActivationLocationType.RESID_POST_MLP,
                layer_index=layer_index,
                tensor_indices=(),
                pass_type=PassType.FORWARD,
            )
        case DerivedScalarType.RESID_POST_ATTN:
            return ActivationIndex(
                activation_location_type=ActivationLocationType.RESID_POST_ATTN,
                layer_index=layer_index,
                tensor_indices=(),
                pass_type=PassType.FORWARD,
            )
        case _:
            raise NotImplementedError(dst)
