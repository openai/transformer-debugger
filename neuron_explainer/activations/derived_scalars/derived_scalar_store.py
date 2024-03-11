"""
This file contains infra for storing multiple ActivationsAndMetadata objects, and computing derived
scalars from them.

The DerivedScalarStore supports functionality like applying a transformation to all the
ActivationsAndMetadata objects within it (for example, computing a max across the token dimension,
or computing the absolute value). It also supports functionality like computing the topk activations
across all the ActivationsAndMetadata objects within it, and returning the corresponding
DerivedScalarIndex objects.
"""

from dataclasses import asdict
from typing import Any, Callable

import blobfile as bf
import numpy as np
import torch

from neuron_explainer.activations.derived_scalars.activations_and_metadata import (
    ActivationsAndMetadata,
    PassType,
    RawActivationStore,
    safe_sorted,
)
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    AttentionTraceType,
    DerivedScalarIndex,
    MirroredActivationIndex,
    MirroredNodeIndex,
    NodeIndex,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import ScalarDeriver
from neuron_explainer.models.model_component_registry import Dimension, LayerIndex, NodeType
from neuron_explainer.pydantic import CamelCaseBaseModel, immutable


@immutable
class AblationSpec(CamelCaseBaseModel):
    """A specification for performing ablation on a model."""

    index: MirroredActivationIndex
    value: float


@immutable
class NodeAblation(CamelCaseBaseModel):
    """A specification for tracing an upstream node.

    This data structure is used by the client. The server converts it to an AblationSpec.
    """

    node_index: MirroredNodeIndex
    value: float


def get_topk_of_tensor_list(
    tensor_list: list[torch.Tensor],
) -> tuple[torch.Tensor, list[int], list[tuple[int, ...]]]:
    """Given a list of tensors, returns the top k values and their indices (within the overall list and within each tensor)
    across all tensors in the list.
    """

    flattened_tensor = torch.cat([tensor.flatten() for tensor in tensor_list]).flatten()
    topk_values, topk_flat_indices = torch.topk(flattened_tensor, k=flattened_tensor.shape[0])
    topk_flat_indices = topk_flat_indices.cpu().numpy()
    cumsum_list_lengths = np.cumsum([tensor.numel() for tensor in tensor_list])

    def convert_flat_index_to_list_and_tensor_indices(
        flat_index: int,
    ) -> tuple[int, tuple[int, ...]]:
        list_index = np.searchsorted(cumsum_list_lengths, flat_index, side="right")
        if list_index > 0:
            flat_index -= cumsum_list_lengths[list_index - 1]
        tensor_indices = tuple(
            int(index) for index in np.unravel_index(flat_index, tensor_list[list_index].shape)
        )  # cast from np.int64 to int
        return (
            list_index,
            tensor_indices,  # type: ignore
        )

    list_indices_tuple, tensor_indices_tuple = zip(
        *[
            convert_flat_index_to_list_and_tensor_indices(flat_index)
            for flat_index in topk_flat_indices
        ]
    )

    return (
        topk_values,
        list(list_indices_tuple),
        list(tensor_indices_tuple),
    )


@immutable
class UpstreamNodeToTrace(CamelCaseBaseModel):
    """A specification for tracing an upstream node.

    This data structure is used by the client. The server converts it to an activation index and
    an ablation spec.
    """

    node_index: MirroredNodeIndex
    attention_trace_type: AttentionTraceType | None


class DerivedScalarStore:
    """
    An object holding one or more types of derived scalars, providing convenience methods for indexing and performing
    common computations, such as top k activations. Note that this class is passed derived scalars at initialization,
    and is not intended to derive them in the first place, but can apply simple transformations to them.
    """

    activations_and_metadata_by_dst_and_pass_type: dict[
        tuple[DerivedScalarType, PassType], ActivationsAndMetadata
    ]

    def __init__(
        self,
        activations_and_metadata_by_dst_and_pass_type: dict[
            tuple[DerivedScalarType, PassType], ActivationsAndMetadata
        ],
        # this contains a dict of different derived scalars, keyed by their derived scalar type and pass type
    ):
        self.activations_and_metadata_by_dst_and_pass_type = (
            activations_and_metadata_by_dst_and_pass_type
        )
        self.sorted_layer_indices_by_dst_and_pass_type = {
            (dst, pass_type): safe_sorted(
                list(activations_and_metadata.activations_by_layer_index.keys())
            )
            for (
                dst,
                pass_type,
            ), activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.items()
        }

        # assert that everything is on the same device
        for activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.values():
            assert activations_and_metadata.device in [
                self.device,
                None,
            ], f"Device mismatch detected: {self.device=} {activations_and_metadata.device=}"

    @classmethod
    def from_list(
        cls, activations_and_metadata_list: list[ActivationsAndMetadata]
    ) -> "DerivedScalarStore":
        dst_and_pass_types = [
            (activations_and_metadata.dst, activations_and_metadata.pass_type)
            for activations_and_metadata in activations_and_metadata_list
        ]
        assert len(set(dst_and_pass_types)) == len(
            dst_and_pass_types
        ), "All ActivationsAndMetadata must have unique dst"
        activations_and_metadata_by_dst_and_pass_type = {
            (
                activations_and_metadata.dst,
                activations_and_metadata.pass_type,
            ): activations_and_metadata
            for activations_and_metadata in activations_and_metadata_list
        }  # activations_and_metadata objects have shape: dict by layer index, tensor: n_tokens, n_neurons or n_tokens, n_tokens, n_attn_heads
        return cls(activations_and_metadata_by_dst_and_pass_type)

    @property
    def device(self) -> torch.device | None:
        # any of the activations_and_metadata that have a non-None device are constrained to share a device
        # (see __init__). If any of the activations_and_metadata have a non-None device, this returns that device;
        # else it returns None
        device: torch.device | None = None
        for activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.values():
            sub_device = activations_and_metadata.device
            if sub_device is not None:
                device = sub_device
                break
        return device

    def topk(
        self,
        k: int | None,
        pass_type: PassType = PassType.FORWARD,
        dsts: list[DerivedScalarType] | None = None,
        layer_indices: list[int] | None = None,
        largest: bool = True,
    ) -> tuple[torch.Tensor, list[DerivedScalarIndex]]:
        """The strategy here, similar to the topk method of ActivationsAndMetadata, is to compute
        topk for each DST separately, and then combine them. This avoids instantiating a big tensor
        containing the entire contents of the DerivedScalarStore at any point."""
        preprocessed = self
        if layer_indices is not None:
            preprocessed = preprocessed.filter_layers(layer_indices)
        if dsts is not None:
            preprocessed = preprocessed.filter_dsts(dsts)
        preprocessed = preprocessed.filter_pass_type(pass_type)
        if k is None:
            k = preprocessed.numel()
        assert k is not None

        topk_by_dst_and_pass_type = {
            (dst, pass_type): self.activations_and_metadata_by_dst_and_pass_type[
                (dst, pass_type)
            ].topk(
                k=k,
                largest=largest,
            )
            for dst in preprocessed.dsts
        }

        (
            overall_topk_values,
            overall_topk_list_indices,
            overall_topk_tensor_indices,
        ) = get_topk_of_tensor_list([topk[0] for topk in topk_by_dst_and_pass_type.values()])

        topk_ds_indices = [topk[1] for topk in topk_by_dst_and_pass_type.values()]

        overall_topk_ds_indices = []
        for list_index, tensor_indices in zip(
            overall_topk_list_indices, overall_topk_tensor_indices
        ):
            assert len(tensor_indices) == 1
            overall_topk_ds_indices.append(topk_ds_indices[list_index][tensor_indices[0]])

        return overall_topk_values, overall_topk_ds_indices

    def sum(
        self,
        node_type: NodeType | None = None,
        layer_indices: list[int] | None = None,
        dims_to_keep: list[Dimension] | None = None,
    ) -> torch.Tensor:
        # to run this function, need to have the shapes as expected (e.g. can't have previously run transform functions
        # that change shapes of activations)
        self._check_activation_ndims()
        filtered = self
        if node_type is not None:
            filtered = filtered.filter_node_types(node_type)
        if layer_indices is not None:
            filtered = filtered.filter_layers(layer_indices)
        if dims_to_keep is None:
            sum_function = torch.sum
        else:
            dsts = filtered.dsts
            shape_specs = [dst.shape_spec_per_token_sequence for dst in dsts]
            shape_spec = shape_specs[0]
            assert all(
                shape_spec == shape_spec for shape_spec in shape_specs[1:]
            ), f"Expected all shape specs to be the same, but got {shape_specs}"
            assert all(dim in shape_specs[0] for dim in dims_to_keep)
            int_dims_to_keep = {shape_spec.index(dim) for dim in dims_to_keep}
            assert len(int_dims_to_keep) == len(dims_to_keep)
            int_dims_to_discard = [
                dim for dim in range(len(shape_spec)) if dim not in int_dims_to_keep
            ]
            assert len(int_dims_to_discard) == len(shape_spec) - len(dims_to_keep)

            def sum_function(tensor: torch.Tensor) -> torch.Tensor:  # type: ignore
                if len(int_dims_to_discard) == 0:
                    # torch.sum(x, dim=[]) is the same as torch.sum(x, dim=None); our desired behavior is to sum nothing
                    return tensor
                else:
                    summed = torch.sum(tensor, dim=int_dims_to_discard)
                    assert summed.ndim == len(dims_to_keep)
                    return summed

        each_activation_summed = filtered.apply_transform_fn_to_activations(sum_function)
        total: torch.Tensor = 0.0  # type: ignore
        for (
            activations_and_metadata
        ) in each_activation_summed.activations_and_metadata_by_dst_and_pass_type.values():
            for (
                activation_total_for_layer
            ) in activations_and_metadata.activations_by_layer_index.values():
                if dims_to_keep is not None:
                    assert activation_total_for_layer.ndim == len(dims_to_keep), (
                        activation_total_for_layer.shape,
                        dims_to_keep,
                    )
                total += activation_total_for_layer
        if dims_to_keep is not None:
            assert total.ndim == len(dims_to_keep)
        return total

    def sum_abs(
        self,
        node_type: NodeType | None = None,
        layer_indices: list[int] | None = None,
        dims_to_keep: list[Dimension] | None = None,
    ) -> torch.Tensor:
        # convenience function for summing the absolute value of all activations
        return self.apply_transform_fn_to_activations(torch.abs).sum(
            node_type=node_type, layer_indices=layer_indices, dims_to_keep=dims_to_keep
        )

    def max(
        self,
        node_type: NodeType | None = None,
        layer_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, DerivedScalarIndex]:
        # convenience function for computing the maximum value of all activations
        if node_type is None:
            dsts = None
        else:
            dsts = [dst for dst in self.dsts if dst.node_type == node_type]

        values, indices = self.topk(
            k=1,
            largest=True,
            pass_type=PassType.FORWARD,
            dsts=dsts,
            layer_indices=layer_indices,
        )
        return values[0], indices[0]

    def max_abs(
        self,
        node_type: NodeType | None = None,
        layer_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, DerivedScalarIndex]:
        # convenience function for computing the maximum absolute value of all activations
        return self.apply_transform_fn_to_activations(torch.abs).max(
            node_type=node_type, layer_indices=layer_indices
        )

    def num_nonzero(
        self,
        node_type: NodeType | None = None,
        layer_indices: list[int] | None = None,
        dims_to_keep: list[Dimension] | None = None,
    ) -> torch.Tensor:
        # convenience function for counting the number of nonzero activations
        return self.apply_transform_fn_to_activations(lambda x: (x != 0).float()).sum(
            node_type=node_type, layer_indices=layer_indices, dims_to_keep=dims_to_keep
        )

    def numel(
        self,
    ) -> int:
        # convenience function for counting the number of nonzero activations
        running_total = 0
        for activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.values():
            running_total += activations_and_metadata.numel()
        return running_total

    def apply_transform_fn_to_activations(
        self, transform_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> "DerivedScalarStore":
        return DerivedScalarStore(
            {
                (
                    dst,
                    pass_type,
                ): activations_and_metadata.apply_transform_fn_to_activations(
                    transform_fn,
                    output_dst=dst,
                    output_pass_type=pass_type,
                )
                for (
                    dst,
                    pass_type,
                ), activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.items()
            }
        )

    def apply_transform_fn_to_multiple_activations(
        self,
        transform_fn: Callable[..., torch.Tensor],
        others: tuple["DerivedScalarStore", ...],
    ) -> "DerivedScalarStore":
        for other in others:
            assert set(self.activations_and_metadata_by_dst_and_pass_type.keys()) == set(
                other.activations_and_metadata_by_dst_and_pass_type.keys()
            )
        return DerivedScalarStore(
            {
                (
                    dst,
                    pass_type,
                ): activations_and_metadata.apply_transform_fn_to_multiple_activations(
                    transform_fn,
                    others=tuple(
                        other.activations_and_metadata_by_dst_and_pass_type[(dst, pass_type)]
                        for other in others
                    ),
                    output_dst=dst,
                    output_pass_type=pass_type,
                )
                for (
                    dst,
                    pass_type,
                ), activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.items()
            }
        )

    def average(
        self,
        others: tuple["DerivedScalarStore", ...],
    ) -> "DerivedScalarStore":
        def average_fn(*args: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.stack(args), dim=0)

        return self.apply_transform_fn_to_multiple_activations(average_fn, others)

    def __add__(self, other: "DerivedScalarStore") -> "DerivedScalarStore":
        def add_fn(*args: torch.Tensor) -> torch.Tensor:
            return torch.sum(torch.stack(args), dim=0)

        return self.apply_transform_fn_to_multiple_activations(add_fn, (other,))

    def __sub__(self, other: "DerivedScalarStore") -> "DerivedScalarStore":
        def sub_fn(*args: torch.Tensor) -> torch.Tensor:
            return torch.sub(args[0], args[1])

        return self.apply_transform_fn_to_multiple_activations(sub_fn, (other,))

    def __eq__(self, other: Any) -> bool:
        """
        note that this uses torch.allclose, rather than checking for precise equality
        """
        if not isinstance(other, DerivedScalarStore):
            return False

        dst_and_pts = self.activations_and_metadata_by_dst_and_pass_type.keys()
        other_dst_and_pts = other.activations_and_metadata_by_dst_and_pass_type.keys()
        if not set(dst_and_pts) == set(other_dst_and_pts):
            return False

        for dst_and_pt in dst_and_pts:
            if (
                not self.activations_and_metadata_by_dst_and_pass_type[dst_and_pt]
                == other.activations_and_metadata_by_dst_and_pass_type[dst_and_pt]
            ):
                return False

        return True

    def filter_with_function(
        self, filter_fn: Callable[[DerivedScalarType, PassType], bool]
    ) -> "DerivedScalarStore":
        return DerivedScalarStore(
            {
                (
                    dst,
                    pass_type,
                ): activations_and_metadata
                for (
                    dst,
                    pass_type,
                ), activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.items()
                if filter_fn(dst, pass_type)
            }
        )

    def filter_dsts(self, dsts: list[DerivedScalarType]) -> "DerivedScalarStore":
        def filter_fn(dst: DerivedScalarType, pass_type: PassType) -> bool:
            return dst in dsts

        return self.filter_with_function(filter_fn)

    def filter_pass_type(self, pass_type: PassType) -> "DerivedScalarStore":
        def filter_fn(dst: DerivedScalarType, pass_type: PassType) -> bool:
            return pass_type == pass_type

        return self.filter_with_function(filter_fn)

    def filter_dst_and_pass_types(
        self, dst_and_pass_types: list[tuple[DerivedScalarType, PassType]]
    ) -> "DerivedScalarStore":
        def filter_fn(dst: DerivedScalarType, pass_type: PassType) -> bool:
            return (dst, pass_type) in dst_and_pass_types

        return self.filter_with_function(filter_fn)

    def filter_node_types(self, node_type: NodeType) -> "DerivedScalarStore":
        def filter_fn(dst: DerivedScalarType, pass_type: PassType) -> bool:
            return dst.node_type == node_type

        return self.filter_with_function(filter_fn)

    def filter_layers(self, layer_indices: list[int] | None) -> "DerivedScalarStore":
        # layer_indices is None means keep all layers
        if layer_indices is None:
            return self
        else:
            return DerivedScalarStore(
                {
                    (
                        dst,
                        pass_type,
                    ): activations_and_metadata.filter_layers(layer_indices)
                    for (
                        dst,
                        pass_type,
                    ), activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.items()
                }
            )

    @property
    def dsts(self) -> set[DerivedScalarType]:
        return {
            dst for dst, _pass_type in self.activations_and_metadata_by_dst_and_pass_type.keys()
        }

    @property
    def node_types(self) -> set[NodeType]:
        return {dst.node_type for dst in self.dsts}

    @property
    def pass_types(self) -> set[PassType]:
        return {
            pass_type
            for _dst, pass_type in self.activations_and_metadata_by_dst_and_pass_type.keys()
        }

    def __getitem__(self, key: DerivedScalarIndex | list[DerivedScalarIndex]) -> torch.Tensor:
        # indexed by index within layer_indices
        if isinstance(key, list):
            items = [self.__getitem__(k) for k in key]
            if all(isinstance(item, torch.Tensor) for item in items):
                return torch.stack(items)
            else:
                assert all(
                    isinstance(item, float) for item in items
                ), f"Expected all items to be torch tensors or floats, but got {items}"
                return torch.tensor(items)
        else:
            layer_indices = self.sorted_layer_indices_by_dst_and_pass_type[(key.dst, key.pass_type)]
            layer_index = key.layer_index

            assert (
                layer_index in layer_indices
            ), f"Layer index {layer_index} not in layer_indices {layer_indices}"
            indices_for_tensor: tuple[slice | int | None, ...] = tuple(
                slice(None) if index is None else index for index in key.tensor_indices
            )
            tensor_for_layer = self.activations_and_metadata_by_dst_and_pass_type[
                (key.dst, key.pass_type)
            ].activations_by_layer_index[layer_index]
            assert key.dst is not None
            assert len(indices_for_tensor) <= tensor_for_layer.ndim, (
                f"Too many indices for tensor of shape {tensor_for_layer.shape} "
                f"and indices {indices_for_tensor}; "
                f"{key.dst=}, {key.pass_type=}, {key.layer_index=}"
            )
            return tensor_for_layer[indices_for_tensor]

    def _check_activation_ndims(self) -> None:
        # ensure that the shapes for the activation tensors are consistent with the shape specs
        # for the derived scalar types
        for (
            dst,
            pass_type,
        ), activations_and_metadata in self.activations_and_metadata_by_dst_and_pass_type.items():
            shape_spec = dst.shape_spec_per_token_sequence
            for activations in activations_and_metadata.activations_by_layer_index.values():
                assert activations.ndim == len(shape_spec), (
                    f"Expected activations to have ndim {len(shape_spec)}, but got {activations.shape=} "
                    f"for {dst=}, {pass_type=}"
                )

    def to_dict(self) -> dict[tuple[str, str], Any]:
        return {
            (dst.value, pt.value): _convert_activations_to_string_keyed_dict(acts.clone())
            for (dst, pt), acts in self.activations_and_metadata_by_dst_and_pass_type.items()
        }

    @classmethod
    def from_dict(
        cls, dict_version: dict[tuple[str, str], Any], skip_missing_dsts: bool
    ) -> "DerivedScalarStore":
        activations_and_metadata_by_dst_and_pass_type: dict[
            tuple[DerivedScalarType, PassType], ActivationsAndMetadata
        ] = {}
        for (dst_value, pt_value), activations_dict in dict_version.items():
            # optionally ignoring missing DSTs (e.g. if we're loading a DerivedScalarStore from before some
            # DerivedScalarTypes were deleted)
            try:
                dst = DerivedScalarType(
                    dst_value
                )  # Enum __init__ is idempotent, so dst_value can be str or DerivedScalarType
            except ValueError:
                if skip_missing_dsts:
                    print(
                        "=" * 30
                        + f"WARNING: SKIPPING MISSING DST {dst_value} AT PASS TYPE {pt_value}"
                        + "=" * 30
                    )
                    continue
                else:
                    raise
            pt = PassType(pt_value)
            assert activations_dict["dst"] == dst.value
            assert activations_dict["pass_type"] == pt.value
            activations_and_metadata_by_dst_and_pass_type[
                (dst, pt)
            ] = _convert_string_keyed_dict_to_activations(activations_dict)
        return cls(activations_and_metadata_by_dst_and_pass_type)

    def save_to_file(self, path: str) -> None:
        with bf.BlobFile(path, "wb") as f:
            torch.save(
                self.to_dict(),
                f,
            )

    @classmethod
    def load_from_file(
        cls, path: str, map_location: torch.device | None = None, skip_missing_dsts: bool = False
    ) -> "DerivedScalarStore":
        with bf.BlobFile(path, "rb") as f:
            serialized_data = torch.load(f, map_location=map_location)
        return cls.from_dict(serialized_data, skip_missing_dsts)

    @classmethod
    def derive_from_raw(
        cls, raw_activation_store: RawActivationStore, scalar_derivers: list[ScalarDeriver]
    ) -> "DerivedScalarStore":
        assert len(scalar_derivers) == len(
            {scalar_deriver.dst for scalar_deriver in scalar_derivers}
        )
        activations_and_metadata_by_dst_and_pass_type: dict[
            tuple[DerivedScalarType, PassType], ActivationsAndMetadata
        ] = {}
        for scalar_deriver in scalar_derivers:
            for pass_type in scalar_deriver.derivable_pass_types:
                activations_and_metadata_by_dst_and_pass_type[
                    (scalar_deriver.dst, pass_type)
                ] = scalar_deriver.derive_from_raw(raw_activation_store, pass_type)
        return cls(activations_and_metadata_by_dst_and_pass_type)

    def get_dst_for_node_type(self, node_type: NodeType) -> DerivedScalarType:
        assert node_type in self.node_types
        dsts = [dst for dst in self.dsts if dst.node_type == node_type]
        assert len(dsts) == 1, f"Expected exactly one DST for node type {node_type}, but got {dsts}"
        return dsts[0]

    def get_ds_indices_for_node_indices(
        self, node_indices: list[NodeIndex]
    ) -> list[DerivedScalarIndex]:
        return [
            DerivedScalarIndex.from_node_index(
                node_index, self.get_dst_for_node_type(node_index.node_type)
            )
            for node_index in node_indices
        ]

    def get_derived_scalars_for_node_indices(self, node_indices: list[NodeIndex]) -> torch.Tensor:
        return self[self.get_ds_indices_for_node_indices(node_indices)]


def _convert_activations_to_string_keyed_dict(
    activations_and_metadata: ActivationsAndMetadata,
) -> dict[str, Any]:
    dict_version = asdict(activations_and_metadata)
    dict_version["dst"] = activations_and_metadata.dst.value
    dict_version["pass_type"] = activations_and_metadata.pass_type.value
    return dict_version


def _convert_string_keyed_dict_to_activations(
    dict_version: dict[str, Any],
) -> ActivationsAndMetadata:
    dict_version["dst"] = DerivedScalarType(dict_version["dst"])
    dict_version["pass_type"] = PassType(dict_version["pass_type"])
    return ActivationsAndMetadata(**dict_version)
