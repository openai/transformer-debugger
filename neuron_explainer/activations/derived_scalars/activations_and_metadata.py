"""
RawActivationStore collects raw activations in ActivationsAndMetadata objects, associated with the
location type and pass type of that raw activation.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import DerivedScalarIndex
from neuron_explainer.activations.derived_scalars.locations import LayerIndexer
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    ActivationLocationTypeAndPassType,
    LayerIndex,
    PassType,
)

LayerwiseTransformFn = Callable[[torch.Tensor, LayerIndex], torch.Tensor]
MultiArgLayerwiseTransformFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ActivationsAndMetadata:
    """Contains data about the internal state of the network during inference, indexed by layer."""

    dst: DerivedScalarType
    pass_type: PassType
    activations_by_layer_index: dict[LayerIndex, torch.Tensor]  # layer index is None iff

    def __post_init__(self) -> None:
        # assert that all the devices are the same or None
        device = self.device
        if len(self.activations_by_layer_index) > 0:
            for activations in self.activations_by_layer_index.values():
                if activations.device is not None:
                    assert activations.device == device

    @property
    def layer_indices(self) -> list[LayerIndex]:
        return safe_sorted(list(self.activations_by_layer_index.keys()))

    @property
    def device(self) -> torch.device | None:
        if len(self.activations_by_layer_index) > 0:
            return next(iter(self.activations_by_layer_index.values())).device
        else:
            # if there are no activations in this object (e.g. if performing a backward pass
            # from a layer 0 attention head, and examining MLP activations) then there are no
            # tensors and thus no devices
            return None

    def cpu(self) -> "ActivationsAndMetadata":
        """Move the activations tensors to cpu."""
        return dataclasses.replace(
            self,
            activations_by_layer_index=_move_tensor_to_cpu_by_layer_index(
                self.activations_by_layer_index
            ),
        )

    def clone(self) -> "ActivationsAndMetadata":
        """Clone the activations tensors."""
        return dataclasses.replace(
            self,
            activations_by_layer_index=_clone_tensor_by_layer_index(
                self.activations_by_layer_index
            ),
        )

    def _remap_layer_indices(
        self, source_layer_index_by_layer_index: dict[LayerIndex, LayerIndex | Literal["Dummy"]]
    ) -> "ActivationsAndMetadata":
        """ending_layer_indices specifies the new layer_index to which each layer_index in self.layer_indices is to be assigned,
        The resulting ActivationsAndMetadata object will have the same layer_indices as the starting object, with undefined layer_indices filled in with
        dummy (empty) tensors.
        See the docstring of apply_layer_indexer for more information.
        """
        activations_by_layer_index = _remap_tensor_by_layer_index(
            self.activations_by_layer_index, source_layer_index_by_layer_index
        )

        return dataclasses.replace(
            self,
            activations_by_layer_index=activations_by_layer_index,
        )

    def apply_layer_indexer(
        self,
        layer_indexer: LayerIndexer,
        desired_layer_indices: (
            list[LayerIndex] | None
        ) = None,  # indicates layer indices to keep; None indicates keep all
    ) -> "ActivationsAndMetadata":
        """DSTs can require an activation from an arbitrary layer (e.g. the n'th layer) and an activation from a different (e.g. constant or offset) layer
        (e.g. the final layer, or the previous layer); for example, MLP activations and the gradient at the final residual stream
        location. The LayerIndexer specifies how to map the layer indices of the activations_by_layer_index in e.g. the residual stream gradient ActivationsAndMetadata object
        to the layer indices of the activations_by_layer_index in the WriteToFinalResidual derived scalar.
        If desired_layer_indices is not None, then the resulting object will have only the layer indices in desired_layer_indices.
        If desired_layer_indices is None, then the resulting object will have the same layer indices as the starting object.
        """
        if desired_layer_indices is None:
            desired_layer_indices = self.layer_indices
        layer_index_source_list = layer_indexer(desired_layer_indices)
        assert len(layer_index_source_list) == len(desired_layer_indices)
        layer_index_source_by_ending_layer_index = dict(
            zip(desired_layer_indices, layer_index_source_list)
        )
        return self._remap_layer_indices(layer_index_source_by_ending_layer_index)

    def apply_transform_fn_to_activations(
        self,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
        output_dst: DerivedScalarType,
        output_pass_type: PassType,
    ) -> "ActivationsAndMetadata":
        """Convenience method to apply the same function to activations_by_layer_index at every
        layer index."""

        def transform_fn_with_unused_layer_index(
            activations: torch.Tensor, _: LayerIndex
        ) -> torch.Tensor:
            return transform_fn(activations)

        return self.apply_layerwise_transform_fn_to_activations(
            transform_fn_with_unused_layer_index,
            output_dst=output_dst,
            output_pass_type=output_pass_type,
        )

    def apply_transform_fn_to_multiple_activations(
        self,
        transform_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        others: tuple["ActivationsAndMetadata", ...],
        output_dst: DerivedScalarType,
        output_pass_type: PassType,
    ) -> "ActivationsAndMetadata":
        """Same as above, but transform_fn takes a second tensor as an argument.
        Its activations_by_layer_index tensors are used as that second argument."""

        def transform_fn_with_unused_layer_index(
            activations: torch.Tensor,
            *other_activations: torch.Tensor,
            layer_index: LayerIndex,
        ) -> torch.Tensor:
            return transform_fn(activations, *other_activations)

        return self.apply_layerwise_transform_fn_to_multiple_activations(
            transform_fn_with_unused_layer_index,
            others=others,
            output_dst=output_dst,
            output_pass_type=output_pass_type,
        )

    # TODO: this function should take transform_fn_with_layer_index, a Callable[[torch.Tensor, LayerIndex], torch.Tensor]
    def apply_layerwise_transform_fn_to_activations(
        self,
        layerwise_transform_fn: LayerwiseTransformFn,
        output_dst: DerivedScalarType,
        output_pass_type: PassType,
    ) -> "ActivationsAndMetadata":
        """layerwise_transform_fn is a Callable[[torch.Tensor, LayerIndex], torch.Tensor]. This function applies the transform_fn
        to the activations_by_layer_index at each layer index, and returns a new ActivationsAndMetadata
        object with the transformed activations."""

        def _layerwise_transform_activations_by_layer_index(
            activations_by_layer_index: dict[LayerIndex, torch.Tensor],
            layerwise_transform_fn: LayerwiseTransformFn,
        ) -> dict[LayerIndex, torch.Tensor]:
            return {
                layer_index: layerwise_transform_fn(activations, layer_index)
                for layer_index, activations in activations_by_layer_index.items()
            }

        return dataclasses.replace(
            self,
            activations_by_layer_index=_layerwise_transform_activations_by_layer_index(
                self.activations_by_layer_index,
                layerwise_transform_fn=layerwise_transform_fn,
            ),
            dst=output_dst,
            pass_type=output_pass_type,
        )

    def apply_layerwise_transform_fn_to_multiple_activations(
        self,
        layerwise_transform_fn: MultiArgLayerwiseTransformFn,  # input to Callable is N tensors
        # this must take a layer_index kwarg, to distinguish it from the arbitrary number of tensor args
        others: tuple["ActivationsAndMetadata", ...],  # N - 1 entries
        output_dst: DerivedScalarType,
        output_pass_type: PassType,
    ) -> "ActivationsAndMetadata":
        """Same as above, but the transform_fn takes
        two tensors as arguments, and the activations_by_layer_index tensors from 'other' are used
        as the second argument."""

        def _layerwise_transform_multiple_activations_by_layer_index(
            activations_by_layer_index: dict[LayerIndex, torch.Tensor],
            layerwise_transform_fn: MultiArgLayerwiseTransformFn,
            other_activations_by_layer_index_tuple: tuple[dict[LayerIndex, torch.Tensor], ...],
        ) -> dict[LayerIndex, torch.Tensor]:
            for other_activations_by_layer_index in other_activations_by_layer_index_tuple:
                assert set(activations_by_layer_index.keys()) == set(
                    other_activations_by_layer_index.keys()
                ), (
                    f"{activations_by_layer_index.keys()=} , "
                    f"{other_activations_by_layer_index.keys()=}"
                )
            return {
                layer_index: layerwise_transform_fn(
                    activations,
                    *[
                        other_activations_by_layer_index[layer_index]
                        for other_activations_by_layer_index in other_activations_by_layer_index_tuple
                    ],
                    layer_index=layer_index,
                )
                for layer_index, activations in activations_by_layer_index.items()
            }

        self_activations_by_layer_index = self.activations_by_layer_index
        other_activations_by_layer_index_tuple = tuple(
            other.activations_by_layer_index for other in others
        )
        for other_activations_by_layer_index in other_activations_by_layer_index_tuple:
            assert set(other_activations_by_layer_index.keys()) == set(
                self_activations_by_layer_index.keys()
            ), (
                f"{other_activations_by_layer_index.keys()=} , "
                f"{self_activations_by_layer_index.keys()=}"
            )

        return dataclasses.replace(
            self,
            activations_by_layer_index=_layerwise_transform_multiple_activations_by_layer_index(
                self_activations_by_layer_index,
                layerwise_transform_fn=layerwise_transform_fn,
                other_activations_by_layer_index_tuple=other_activations_by_layer_index_tuple,
            ),
            dst=output_dst,
            pass_type=output_pass_type,
        )

    def filter_layers(self, layer_indices: list[int] | None) -> "ActivationsAndMetadata":
        """Returns a new ActivationsAndMetadata object with only the specified layer indices."""
        if layer_indices is None:
            return self
        else:
            return dataclasses.replace(
                self,
                activations_by_layer_index={
                    layer_index: activations
                    for layer_index, activations in self.activations_by_layer_index.items()
                    if layer_index in layer_indices or layer_index is None
                },
            )

    @property
    def shape(self) -> tuple[int, ...]:
        first_value = next(iter(self.activations_by_layer_index.values()))
        shape = first_value.shape
        return tuple(shape)

    def __eq__(self, other: Any) -> bool:
        """
        Note that this uses torch.allclose, rather than checking for precise equality.

        This permits ActivationsAndMetadata to be "equal" while having different dst
        and pass type. This is useful for situations where we want to compare two derived scalars
        that should be the same but that are computed in different ways
        """
        if not isinstance(other, ActivationsAndMetadata):
            return False

        def check_activations_by_layer_index_equality(
            self_activations_by_layer_index: dict[LayerIndex, torch.Tensor],
            other_activations_by_layer_index: dict[LayerIndex, torch.Tensor],
        ) -> bool:
            # check indices
            if set(self_activations_by_layer_index.keys()) != set(
                other_activations_by_layer_index.keys()
            ):
                return False
            # check shapes and then values
            for layer_index in self_activations_by_layer_index.keys():
                if (
                    self_activations_by_layer_index[layer_index].shape
                    != other_activations_by_layer_index[layer_index].shape
                ):
                    return False
                if not torch.allclose(
                    self_activations_by_layer_index[layer_index],
                    other_activations_by_layer_index[layer_index],
                ):
                    return False
            return True

        if not check_activations_by_layer_index_equality(
            self.activations_by_layer_index, other.activations_by_layer_index
        ):
            return False

        return True

    def __add__(self, other: "ActivationsAndMetadata") -> "ActivationsAndMetadata":
        def add_fn(*args: torch.Tensor) -> torch.Tensor:
            return torch.sum(torch.stack(args), dim=0)

        return self.apply_transform_fn_to_multiple_activations(
            add_fn, (other,), output_dst=self.dst, output_pass_type=self.pass_type
        )

    def __sub__(self, other: "ActivationsAndMetadata") -> "ActivationsAndMetadata":
        def sub_fn(*args: torch.Tensor) -> torch.Tensor:
            return torch.sub(args[0], args[1])

        return self.apply_transform_fn_to_multiple_activations(
            sub_fn, (other,), output_dst=self.dst, output_pass_type=self.pass_type
        )

    def max(self) -> tuple[torch.Tensor, DerivedScalarIndex]:
        values, indices = self.topk(1, largest=True)
        return values[0], indices[0]

    def sum(self) -> torch.Tensor:
        return torch.sum(torch.stack(list(self.activations_by_layer_index.values())))

    def sum_abs(self) -> torch.Tensor:
        return torch.sum(torch.abs(torch.stack(list(self.activations_by_layer_index.values()))))

    def topk(self, k: int, largest: bool) -> tuple[torch.Tensor, list[DerivedScalarIndex]]:
        # this first computes topk values and indices for each layer, then stacks them and computes topk values and indices
        # the topk for the overall stack. This avoids instantiating a second copy of all the data
        # in self.activations_by_layer_index
        if k > self.numel():
            k = self.numel()  # if k > numel is requested, return everything

        def get_topk_indices(activations: torch.Tensor) -> torch.Tensor:
            if k >= activations.numel():
                return torch.argsort(activations.flatten(), descending=largest)
            else:
                _, indices = torch.topk(activations.flatten(), k, largest=largest)
            return indices

        def get_topk_values(
            activations: torch.Tensor, indices: torch.Tensor, layer_index: LayerIndex
        ) -> torch.Tensor:
            # layer_index is unused, but required as a keyword argument
            return torch.gather(activations.flatten(), 0, indices)

        topk_indices = self.apply_transform_fn_to_activations(
            get_topk_indices, output_dst=self.dst, output_pass_type=self.pass_type
        )
        topk_values = self.apply_layerwise_transform_fn_to_multiple_activations(
            get_topk_values, (topk_indices,), output_dst=self.dst, output_pass_type=self.pass_type
        )

        topk_values_list = []
        for layer_index in self.layer_indices:
            topk_values_list.append(topk_values.activations_by_layer_index[layer_index])
        stacked_topk_values = torch.stack(topk_values_list)

        overall_topk_values, overall_topk_indices = torch.topk(
            stacked_topk_values.flatten(), k, largest=largest
        )

        overall_topk_layer_index_indices, overall_topk_topk_indices = np.unravel_index(
            overall_topk_indices.cpu().numpy(), stacked_topk_values.shape
        )

        overall_topk_layer_indices = [
            self.layer_indices[i] for i in overall_topk_layer_index_indices
        ]

        overall_topk_ds_indices = [
            DerivedScalarIndex(
                dst=self.dst,
                pass_type=self.pass_type,
                layer_index=layer_index,
                tensor_indices=tuple(
                    int(x)
                    for x in np.unravel_index(
                        int(
                            topk_indices.activations_by_layer_index[layer_index][
                                overall_topk_topk_indices[i]
                            ].item()
                        ),
                        self.activations_by_layer_index[layer_index].shape,
                    )
                ),  # cast from np.int64 to int
            )
            for i, layer_index in enumerate(overall_topk_layer_indices)
        ]

        return overall_topk_values, overall_topk_ds_indices

    def numel(self) -> int:
        return sum(activations.numel() for activations in self.activations_by_layer_index.values())


def _fill_in_activations_by_layer_index_at_layer_indices(
    activations_by_layer_index: dict[LayerIndex, torch.Tensor], layer_indices: set[LayerIndex]
) -> dict[LayerIndex, torch.Tensor]:
    """
    some activations might not be filled in to start, e.g. if they are from a layer after the point
    from which a backward pass was computed. In this case, missing values in activations_by_layer_index.values()
    are filled in to be zero tensors. This is well motivated because dUpstreamActivation/dDownstreamActivation is
    in fact 0.
    """
    if None in activations_by_layer_index:
        assert len(activations_by_layer_index) == 1
        default_tensor = activations_by_layer_index.pop(None)
    else:
        example_tensor = next(iter(activations_by_layer_index.values()))
        default_tensor = torch.zeros_like(example_tensor)
    for layer_index in layer_indices:
        if layer_index not in activations_by_layer_index:
            activations_by_layer_index[layer_index] = default_tensor
    return activations_by_layer_index


def _remap_tensor_by_layer_index(
    starting_tensor_by_layer_index: dict[LayerIndex, torch.Tensor],
    source_layer_index_by_layer_index: dict[LayerIndex, LayerIndex | Literal["Dummy"]],
) -> dict[LayerIndex, torch.Tensor]:
    # TODO: clarify comment to indicate "Dummy" tensors can be either "truly 0" tensors, as in the case of backward passes, or "invalid"
    # tensors, as in the case of referring to a layer that does not exist.
    """source_layer_index_by_layer_index specifies the starting layer_index (value) from which each new layer_index (key) should get its tensor,
    The resulting dict of tensors will have the same layer_indices as the starting dict, with undefined layer_indices filled in with
    zero tensors. Used when shifting layer indices of ActivationsAndMetadata objects, prior to applying some downstream transformation to them.
    "Dummy" is used to indicate that the corresponding activations are somehow 'invalid', e.g. coming from a previous layer when there is no
    previous layer.
    Callers will look for all the same activations in each layer, even though some activations in some layers will not be used.
    Therefore, we use dummy activations to pass to those callers, with the floats within those activations not affecting the output of
    the callers."""

    if len(starting_tensor_by_layer_index) == 0:
        return {
            layer_index: torch.tensor(0.0)
            for layer_index in source_layer_index_by_layer_index.keys()
        }
    example_tensor = next(iter(starting_tensor_by_layer_index.values()))
    device = example_tensor.device
    dummy_tensor = torch.tensor(0.0, device=device)  # maybe change to have shape of example_tensor

    # fill in missing starting indices with default values (empty tensors)
    non_dummy_source_layer_indices = [
        source_layer_index
        for source_layer_index in list(source_layer_index_by_layer_index.values())
        if source_layer_index != "Dummy"
    ]
    starting_tensor_by_layer_index = _fill_in_activations_by_layer_index_at_layer_indices(
        starting_tensor_by_layer_index, set(non_dummy_source_layer_indices)
    )

    def _get_value_or_dummy(
        starting_tensor_by_layer_index: dict[LayerIndex, torch.Tensor],
        source_layer_index_or_dummy: LayerIndex | Literal["Dummy"],
    ) -> torch.Tensor:
        if source_layer_index_or_dummy == "Dummy":
            return dummy_tensor
        else:
            assert (
                source_layer_index_or_dummy in starting_tensor_by_layer_index
            ), f"{source_layer_index_or_dummy=} , {starting_tensor_by_layer_index.keys()=}"
            return starting_tensor_by_layer_index[source_layer_index_or_dummy]

    # fill in ending_layer_indices with the corresponding starting_layer_index
    return {
        layer_index: _get_value_or_dummy(starting_tensor_by_layer_index, source_layer_index)
        for layer_index, source_layer_index in source_layer_index_by_layer_index.items()
    }


def _clone_tensor_by_layer_index(
    tensor_by_layer_index: dict[LayerIndex, torch.Tensor]
) -> dict[LayerIndex, torch.Tensor]:
    return {
        # In addition to cloning, we detach. It's hard to imagine a situation where we'd want the
        # cloned tensor to remain part of the computation graph.
        layer_index: tensor.detach().clone()
        for layer_index, tensor in tensor_by_layer_index.items()
    }


def _move_tensor_to_cpu_by_layer_index(
    tensor_by_layer_index: dict[LayerIndex, torch.Tensor]
) -> dict[LayerIndex, torch.Tensor]:
    return {layer_index: tensor.cpu() for layer_index, tensor in tensor_by_layer_index.items()}


def safe_sorted(x: list[LayerIndex]) -> list[LayerIndex]:
    if len(x) > 1:
        # a list of integer layer indices, e.g. [0, 1, 2, 5] (for location types with layer indices, like MLP activations)
        assert all(element is not None for element in x)
        return sorted(x)  # type: ignore
    else:
        # a single list with either [None] (for location types with no layer indices, like embeddings)
        # or a single layer index (for location types with layer indices, like MLP activations)
        return x


@dataclass(frozen=True)
class RawActivationStore:
    """
    Holds activations of multiple HookLocationTypeAndPassType's, and computes derived scalars from them given one or
    more scalar derivers and requested pass types. This class is intended to be used for computing derived scalars.
    """

    activations_by_sub_activation_location_type_and_pass_type: dict[
        ActivationLocationTypeAndPassType, ActivationsAndMetadata
    ]

    def get_activations_and_metadata(
        self,
        activation_location_type: ActivationLocationType,
        pass_type: PassType,
    ) -> ActivationsAndMetadata:
        return self.activations_by_sub_activation_location_type_and_pass_type[
            ActivationLocationTypeAndPassType(
                activation_location_type=activation_location_type, pass_type=pass_type
            )
        ]

    @classmethod
    def from_nested_dict_of_activations(
        cls,
        activations_by_sub_activation_location_type_and_pass_type: dict[
            ActivationLocationTypeAndPassType, dict[LayerIndex, torch.Tensor]
        ],
    ) -> "RawActivationStore":
        """This automatically constructs ActivationsAndMetadata objects from dicts,
        as well as the overall RawActivationStore object from the resulting ActivationsAndMetadata objects.
        """
        activations_and_metadata_by_activation_location_type_and_pass_type = {}
        for (
            activation_location_type_and_pass_type,
            activations_by_layer_index,
        ) in activations_by_sub_activation_location_type_and_pass_type.items():
            dst = DerivedScalarType.from_activation_location_type(
                activation_location_type_and_pass_type.activation_location_type
            )
            activations_and_metadata = ActivationsAndMetadata(
                activations_by_layer_index=activations_by_layer_index,
                dst=dst,
                pass_type=activation_location_type_and_pass_type.pass_type,
            )
            activations_and_metadata_by_activation_location_type_and_pass_type[
                activation_location_type_and_pass_type
            ] = activations_and_metadata
        return cls(activations_and_metadata_by_activation_location_type_and_pass_type)
