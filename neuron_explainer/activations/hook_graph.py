"""
This module contains classes for injecting hooks into a Transformer using the
ActivationLocationType and PassType ontology. These produce activation location types that would not
have otherwise existed. For example, the AutoencoderHookGraph is necessary for the
ActivationLocationType.ONLINE_AUTOENCODER_LATENT location to exist.
"""

from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Mapping, cast

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.hooks import (
    AtLayers,
    AutoencoderHooks,
    HookCollection,
    Hooks,
    TransformerHooks,
)
from neuron_explainer.models.inference_engine_type_registry import (
    InferenceEngineType,
    get_hook_location_type_for_activation_location_type,
    standard_model_activation_location_types,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    ActivationLocationTypeAndPassType,
    LayerIndex,
    PassType,
)


def unflatten(f: Callable) -> Callable:
    def _f(x: torch.Tensor) -> torch.Tensor:
        return f(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1], -1)

    return _f


def _append_to_hook_collection_using_string_list(
    hook_collection: HookCollection, string_list: list[str], hook: Callable
) -> None:
    assert len(string_list) > 0
    assert (
        string_list[0] in hook_collection.all_hooks
    ), f"string_list: {string_list}, hook_collection: {hook_collection}"

    sub_hook_collection = hook_collection.all_hooks[string_list[0]]

    if len(string_list) == 1:
        assert isinstance(
            sub_hook_collection, Hooks
        ), f"string_list: {string_list}, hook_collection: {type(hook_collection)}, sub_hook_collection: {type(sub_hook_collection)}"
        sub_hook_collection.append(hook)
    else:
        assert isinstance(
            sub_hook_collection, HookCollection
        ), f"string_list: {string_list}, hook_collection: {type(hook_collection)}, sub_hook_collection: {type(sub_hook_collection)}"
        _append_to_hook_collection_using_string_list(sub_hook_collection, string_list[1:], hook)


def _append_to_hook_collection_using_activation_location_type_and_pass_type(
    hook_collection: HookCollection,
    activation_location_type_and_pass_type: ActivationLocationTypeAndPassType,
    hook: Callable,
    append_to_fwd2: bool = False,
) -> None:
    activation_location_type = activation_location_type_and_pass_type.activation_location_type
    pass_type = activation_location_type_and_pass_type.pass_type

    standard_model_hook_location_type = get_hook_location_type_for_activation_location_type(
        activation_location_type, inference_engine_type=InferenceEngineType.STANDARD
    )

    if (
        "resid" in standard_model_hook_location_type
        and "post_emb" not in standard_model_hook_location_type
        and "ln_f" not in standard_model_hook_location_type
        and "post_ln_f" not in standard_model_hook_location_type
    ):
        # an extra "torso" is needed for the residual location types
        standard_model_hook_location_type = standard_model_hook_location_type.replace(
            "resid", "resid/torso"
        )

    string_list = standard_model_hook_location_type.split("/")  # e.g. ["mlp", "post_act"]
    if append_to_fwd2:
        assert pass_type == PassType.FORWARD
        string_list += ["fwd2"]  # called after all "fwd" and "bwd" hooks
    else:
        string_list += [_pass_type_hc_name_by_hook_pass_type[pass_type]]

    _append_to_hook_collection_using_string_list(hook_collection, string_list, hook)


_pass_type_hc_name_by_hook_pass_type: dict[PassType, str] = {
    PassType.FORWARD: "fwd",
    PassType.BACKWARD: "bwd",
}


class PerLayerHookCollection(HookCollection):
    """
    Organizes HookCollections by layer; supports e.g. appending to the same location within
    each per-layer HookCollection by supplying a callable to apply_fn_to_all_layers, to do that
    appending.
    """

    def __init__(self, hook_collection_by_layer: Mapping[LayerIndex, HookCollection]) -> None:
        super().__init__()

        for layer in hook_collection_by_layer.keys():
            self.add_subhooks(layer, hook_collection_by_layer[layer])

    def __call__(self, x: torch.Tensor, *, layer: LayerIndex = None, **kwargs: Any) -> torch.Tensor:
        if layer in self.all_hooks:
            return self.all_hooks[layer](x, layer=layer, **kwargs)
        else:
            return x

    def append_to_all_layers_using_string_list(
        self, string_list: list[str], hook: Callable
    ) -> None:
        for layer in self.all_hooks.keys():
            _append_to_hook_collection_using_string_list(self.all_hooks[layer], string_list, hook)

    def __deepcopy__(self, memo: dict) -> "PerLayerHookCollection":
        # can't use deepcopy because of __getattr__
        hook_collection_by_layer = self.all_hooks
        new = self.__class__(self.all_hooks)
        new.all_hooks = deepcopy(self.all_hooks)
        return new


class HookGraph(ABC):
    """
    This is a wrapper for HookCollection objects that supports
    1. adding hooks at points specified using activation_location_type + pass_type and optionally layer_indices
    2. adding subgraphs that are themselves HookGraphs, in such a way that activation_location_types within the
    subgraph remain accessible by the same activation_location_type + pass_type + layer_indices interface
    """

    hook_collection: HookCollection
    activation_locations: set[ActivationLocationType]
    subgraph_by_name: dict[str, "InjectableHookGraph"]

    def __call__(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.hook_collection(x, *args, **kwargs)  # type: ignore

    def append(
        self,
        activation_location_type_and_pass_type: ActivationLocationTypeAndPassType,
        hook: Callable,
        layer_indices: int | list[int] | None = None,
        append_to_fwd2: bool = False,
    ) -> None:
        pass

    def inject_subgraph(
        self,
        # activation_location_type_and_pass_type: ActivationLocationTypeAndPassType,
        subgraph: "InjectableHookGraph",
        name: str,
        layer_indices: int | list[int] | None = None,
    ) -> None:
        activation_location_type_and_pass_type = subgraph.at_activation_location_type_and_pass_type
        activation_location_type = activation_location_type_and_pass_type.activation_location_type
        pass_type = activation_location_type_and_pass_type.pass_type

        assert (
            activation_location_type in self.activation_locations
        ), f"{activation_location_type} not in {self.activation_locations}"
        assert name not in self.subgraph_by_name
        # assert no overlap between activation locations of self and graph
        assert not self.activation_locations.intersection(subgraph.activation_locations), (
            self.activation_locations,
            subgraph.activation_locations,
        )
        self.append(
            activation_location_type_and_pass_type=activation_location_type_and_pass_type,
            hook=subgraph,
            layer_indices=layer_indices,
            append_to_fwd2=True,  # we inject the subgraph after the forward and backward hooks
        )
        self.subgraph_by_name[name] = subgraph
        self.activation_locations = self.activation_locations.union(subgraph.activation_locations)


class InjectableHookGraph(HookGraph):
    """
    This is a HookGraph that can be injected into another HookGraph. It contains one extra piece
    of information: the activation_location_type_and_pass_type where it is to be injected.
    """

    at_activation_location_type_and_pass_type: ActivationLocationTypeAndPassType


class TransformerHookGraph(HookGraph):
    """
    This is a HookGraph that specifically wraps TransformerHooks. It can be used with the Transformer.forward()
    function call using the transformer_graph.as_transformer_hooks() method.
    """

    def __init__(self) -> None:
        self.hook_collection = TransformerHooks()
        self.subgraph_by_name: dict[str, InjectableHookGraph] = {}
        self.activation_locations = standard_model_activation_location_types

    def append(
        self,
        activation_location_type_and_pass_type: ActivationLocationTypeAndPassType,
        hook: Callable,
        layer_indices: int | list[int] | None = None,
        append_to_fwd2: bool = False,
    ) -> None:
        activation_location_type = activation_location_type_and_pass_type.activation_location_type
        pass_type = activation_location_type_and_pass_type.pass_type

        if layer_indices is not None:
            assert (
                not activation_location_type.has_no_layers
            ), f"activation_location_type: {activation_location_type}, layer_indices: {layer_indices}"
            hook = AtLayers(layer_indices).append(hook)

        assert (
            activation_location_type in self.activation_locations
        ), f"{activation_location_type} not in {self.activation_locations}"

        if activation_location_type in standard_model_activation_location_types:
            _append_to_hook_collection_using_activation_location_type_and_pass_type(
                self.hook_collection,
                activation_location_type_and_pass_type,
                hook,
                append_to_fwd2,
            )
        else:
            for name in self.subgraph_by_name.keys():
                if activation_location_type in self.subgraph_by_name[name].activation_locations:
                    self.subgraph_by_name[name].append(activation_location_type_and_pass_type, hook)

    def as_transformer_hooks(self) -> TransformerHooks:
        return cast(TransformerHooks, self.hook_collection)


class AutoencoderHookGraph(InjectableHookGraph):
    """
    This is a HookGraph that specifically wraps a PerLayerHookCollection of AutoencoderHooks (in general, one per layer).
    """

    def __init__(
        self, autoencoder_context: AutoencoderContext, is_one_of_multiple_autoencoders: bool = False
    ) -> None:
        autoencoder_hooks_by_layer_index: dict[LayerIndex, AutoencoderHooks] = {}
        layer_indices = autoencoder_context.layer_indices or [None]
        for layer_index in layer_indices:
            autoencoder = autoencoder_context.get_autoencoder(layer_index)
            autoencoder_hooks_by_layer_index[layer_index] = AutoencoderHooks(
                encode=unflatten(autoencoder.encode),
                decode=unflatten(autoencoder.decode),
                add_error=True,
            )
        if not autoencoder_context.dst.is_raw_activation_type:
            raise NotImplementedError(
                "AutoencoderHookGraph only supports raw activation types for now."
            )
        self.at_activation_location_type_and_pass_type = ActivationLocationTypeAndPassType(
            autoencoder_context.dst.to_activation_location_type(), PassType.FORWARD
        )
        self.hook_collection = PerLayerHookCollection(autoencoder_hooks_by_layer_index)
        self.location_hc_name_by_activation_location_type = (
            self.get_location_hc_name_by_activation_location_type(
                autoencoder_context.dst, is_one_of_multiple_autoencoders
            )
        )
        self.activation_locations = set(self.location_hc_name_by_activation_location_type.keys())
        self.autoencoder_context = autoencoder_context

    def append(
        self,
        activation_location_type_and_pass_type: ActivationLocationTypeAndPassType,
        hook: Callable,
        layer_indices: int | list[int] | None = None,
        append_to_fwd2: bool = False,
    ) -> None:
        activation_location_type = activation_location_type_and_pass_type.activation_location_type
        pass_type = activation_location_type_and_pass_type.pass_type

        if layer_indices is not None:
            assert (
                not activation_location_type.has_no_layers
            ), f"activation_location_type: {activation_location_type}, layer_indices: {layer_indices}"
            hook = AtLayers(layer_indices).append(hook)

        assert (
            activation_location_type in self.activation_locations
        ), f"{activation_location_type} not in {self.activation_locations}"
        assert pass_type in _pass_type_hc_name_by_hook_pass_type

        string_list = [
            self.location_hc_name_by_activation_location_type[activation_location_type],
            _pass_type_hc_name_by_hook_pass_type[pass_type],
        ]

        self.hook_collection.append_to_all_layers_using_string_list(string_list, hook)

    # note: hc = hook_collection
    def get_location_hc_name_by_activation_location_type(
        self, dst: DerivedScalarType, is_one_of_multiple_autoencoders: bool
    ) -> dict[ActivationLocationType, str]:
        latent_alt_by_dst = {
            DerivedScalarType.MLP_POST_ACT: ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT,
            DerivedScalarType.RESID_DELTA_MLP: ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT,
            DerivedScalarType.RESID_DELTA_ATTN: ActivationLocationType.ONLINE_ATTENTION_AUTOENCODER_LATENT,
        }
        error_alt_by_dst = {
            DerivedScalarType.MLP_POST_ACT: ActivationLocationType.ONLINE_MLP_AUTOENCODER_ERROR,
            DerivedScalarType.RESID_DELTA_MLP: ActivationLocationType.ONLINE_RESIDUAL_MLP_AUTOENCODER_ERROR,
            DerivedScalarType.RESID_DELTA_ATTN: ActivationLocationType.ONLINE_RESIDUAL_ATTENTION_AUTOENCODER_ERROR,
        }
        location_hc_name_by_alt = {
            latent_alt_by_dst[dst]: "latents",
            error_alt_by_dst[dst]: "error",
        }
        if not is_one_of_multiple_autoencoders:
            # if there is only one autoencoder, we also add the "ONLINE_AUTOENCODER_LATENT" location for backward compatibility
            generic_latent_alt = ActivationLocationType.ONLINE_AUTOENCODER_LATENT
            location_hc_name_by_alt[generic_latent_alt] = "latents"

        return location_hc_name_by_alt
