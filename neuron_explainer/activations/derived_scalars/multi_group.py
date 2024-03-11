"""
This file contains code for computing top-k operations on multiple "groups" of derived
scalars, where a group is defined as having comparable units, and each group contains derived
scalars with the same set of NodeTypes. This is used e.g. for determining the elements to display
on a TDB node table.
"""

from typing import Callable

import torch

from neuron_explainer.activation_server.requests_and_responses import GroupId
from neuron_explainer.activations.derived_scalars.activations_and_metadata import RawActivationStore
from neuron_explainer.activations.derived_scalars.config import DstConfig
from neuron_explainer.activations.derived_scalars.derived_scalar_store import DerivedScalarStore
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import NodeIndex
from neuron_explainer.activations.derived_scalars.make_scalar_derivers import make_scalar_deriver
from neuron_explainer.activations.derived_scalars.scalar_deriver import ScalarDeriver
from neuron_explainer.models.model_component_registry import (
    ActivationLocationTypeAndPassType,
    NodeType,
)


class MultiGroupDerivedScalarStore:
    """
    This contains multiple DerivedScalarStores, each of which is taken to have comparable units
    among different derived scalars in the same DerivedScalarStore. For example, MLP_WRITE_NORM
    and ATTN_WRITE_NORM. The groups of derived scalars are associated with GroupIds. This supports
    a topk operation, which
    1. computes the top k derived scalars within each group, and associates each of the top values
    to a NodeIndex
    2. takes the union over all top NodeIndex objects for in any group
    3. computes the derived scalars for each group, for each NodeIndex in that union
    4. returns the list[NodeIndices] and a dict[GroupId, list[derived scalar values]]
    """

    def __init__(
        self,
        derived_scalars_by_group_id: dict[GroupId, DerivedScalarStore],
        exclude_bottom_k_by_group_id: dict[GroupId, bool] | None = None,
    ):
        self.derived_scalars_by_group_id = derived_scalars_by_group_id
        for ds_store in derived_scalars_by_group_id.values():
            assert (
                ds_store.node_types == self.node_types
            )  # all DerivedScalarStores must contain the same node_types
        if exclude_bottom_k_by_group_id is not None:
            assert set(exclude_bottom_k_by_group_id.keys()) == set(
                derived_scalars_by_group_id.keys()
            )
            self.exclude_bottom_k_by_group_id = exclude_bottom_k_by_group_id
        else:
            self.exclude_bottom_k_by_group_id = {
                group_id: False for group_id in derived_scalars_by_group_id.keys()
            }

    def to_single_ds_store(self) -> DerivedScalarStore:
        assert len(self.derived_scalars_by_group_id) == 1
        assert list(self.derived_scalars_by_group_id.keys()) == [GroupId.SINGLETON]
        return list(self.derived_scalars_by_group_id.values())[0]

    @classmethod
    def derive_from_raw(
        cls,
        raw_activation_store: RawActivationStore,
        multi_group_scalar_derivers: "MultiGroupScalarDerivers",
    ) -> "MultiGroupDerivedScalarStore":
        derived_scalars_by_group_id = {
            group_id: DerivedScalarStore.derive_from_raw(raw_activation_store, scalar_derivers)
            for group_id, scalar_derivers in multi_group_scalar_derivers.scalar_derivers_by_group_id.items()
        }
        exclude_bottom_k_by_group_id = multi_group_scalar_derivers.exclude_bottom_k_by_group_id
        return cls(derived_scalars_by_group_id, exclude_bottom_k_by_group_id)

    @property
    def group_ids(self) -> set[GroupId]:
        return set(self.derived_scalars_by_group_id.keys())

    @property
    def node_types(self) -> set[NodeType]:
        return next(iter(self.derived_scalars_by_group_id.values())).node_types

    def get_ds_store(self, group_id: GroupId) -> DerivedScalarStore:
        assert group_id in self.derived_scalars_by_group_id, (
            group_id,
            self.derived_scalars_by_group_id.keys(),
        )
        return self.derived_scalars_by_group_id[group_id]

    def _topk_of_group(
        self,
        group_id: GroupId,
        top_and_bottom_k: int | None,
        transform_activations_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        transform_indices_fn: Callable[[NodeIndex], NodeIndex] | None = None,
    ) -> tuple[torch.Tensor, list[NodeIndex]]:
        """
        Depending on whether self.exclude_bottom_k_by_group_id[group_id] is True or False, this will return either the top k activations
        or the top and bottom k activations, respectively; in the latter case, they are concatenated together
        """
        derived_scalar_store = self.get_ds_store(group_id)
        if transform_activations_fn is not None:
            derived_scalar_store = derived_scalar_store.apply_transform_fn_to_activations(
                transform_activations_fn
            )
        top_activations, top_ds_indices = derived_scalar_store.topk(top_and_bottom_k, largest=True)
        if not self.exclude_bottom_k_by_group_id[group_id]:
            bottom_activations, bottom_ds_indices = derived_scalar_store.topk(
                top_and_bottom_k, largest=False
            )
            top_activations = torch.cat([top_activations, bottom_activations], dim=0)
            top_ds_indices += bottom_ds_indices
        top_node_indices = [NodeIndex.from_ds_index(ds_index) for ds_index in top_ds_indices]
        if transform_indices_fn is not None:
            top_node_indices = [transform_indices_fn(node_index) for node_index in top_node_indices]
        return top_activations, top_node_indices

    def topk(
        self,
        top_and_bottom_k: int | None,
        transform_activations_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        transform_indices_fn: Callable[[NodeIndex], NodeIndex] | None = None,
    ) -> tuple[list[NodeIndex], dict[GroupId, torch.Tensor]]:
        """
        Return all node indices which were in the top or bottom k of any group id (as applicable according to self.exclude_bottom_k_by_group_id)
        Also return the derived scalar values for each group id for each node index, in a dict
        """
        top_node_indices_list = []
        for group_id in self.derived_scalars_by_group_id.keys():
            _, top_node_indices = self._topk_of_group(
                group_id,
                top_and_bottom_k,
                transform_activations_fn,
                transform_indices_fn,
            )
            top_node_indices_list.extend(top_node_indices)

        # Create a list of all node indices across all groups, removing duplicates. We need to maintain
        # a set and a list because we need to maintain the original order (which is why we can't use the
        # list(set()) trick). We need to maintain the original order in order to make the output deterministic.
        # these are the indices of nodes that were in the top-k for at least one kind of derived scalar
        # in parallel construct a matching list of dicts of token information
        all_node_indices = []
        unique_indices = set()
        for node_index in top_node_indices_list:
            if node_index not in unique_indices:
                unique_indices.add(node_index)
                all_node_indices.append(node_index)

        # for nodes that were top-k in at least one kind of derived scalar, convert from the index of the node
        # to the index of the relevant derived scalar for each group id;
        # using the derived scalar indices, access the derived scalar values themselves for all relevant node
        # indices
        activations_by_group_id = self.get_derived_scalars_by_group_id_for_node_indices(
            all_node_indices
        )

        return all_node_indices, activations_by_group_id

    def get_derived_scalars_by_group_id_for_node_indices(
        self, node_indices: list[NodeIndex]
    ) -> dict[GroupId, torch.Tensor]:
        return {
            group_id: (
                self.get_ds_store(group_id).get_derived_scalars_for_node_indices(node_indices)
            )
            for group_id in self.group_ids
        }


class MultiGroupScalarDerivers:
    """
    This contains multiple ScalarDerivers categorized into groups. Each group of ScalarDerivers is taken
    to have outputs with common units (see MultiGroupDerivedScalarStore) docstring
    """

    def __init__(
        self,
        scalar_derivers_by_group_id: dict[GroupId, list[ScalarDeriver]],
        exclude_bottom_k_by_group_id: dict[GroupId, bool] | None = None,
    ):
        self.scalar_derivers_by_group_id = scalar_derivers_by_group_id
        if exclude_bottom_k_by_group_id is not None:
            assert set(exclude_bottom_k_by_group_id.keys()) == set(
                scalar_derivers_by_group_id.keys()
            )
            self.exclude_bottom_k_by_group_id = exclude_bottom_k_by_group_id
        else:
            self.exclude_bottom_k_by_group_id = {
                group_id: False for group_id in scalar_derivers_by_group_id.keys()
            }

    @classmethod
    def from_scalar_derivers(
        cls, scalar_derivers: list[ScalarDeriver]
    ) -> "MultiGroupScalarDerivers":
        return cls({GroupId.SINGLETON: scalar_derivers}, None)

    @classmethod
    def from_dst_and_config_list(
        cls, dst_and_config_list: list[tuple[DerivedScalarType, DstConfig]]
    ) -> "MultiGroupScalarDerivers":
        scalar_derivers = [
            make_scalar_deriver(
                dst=dst,
                dst_config=dst_config,
            )
            for dst, dst_config in dst_and_config_list
        ]
        return cls.from_scalar_derivers(scalar_derivers)

    @classmethod
    def from_dst_and_config_list_by_group_id(
        cls,
        dst_and_config_list_by_group_id: dict[GroupId, list[tuple[DerivedScalarType, DstConfig]]],
        exclude_bottom_k_by_group_id: dict[GroupId, bool] | None = None,
    ) -> "MultiGroupScalarDerivers":
        # NOTE: requires that all groups have the same node types, appearing in the same order within
        # each of dst_and_config_list_by_group_id's values
        node_types_by_group_id = {
            group_id: tuple(dst.node_type for dst, _ in dst_and_config_list)
            for group_id, dst_and_config_list in dst_and_config_list_by_group_id.items()
        }
        assert (
            len(set(node_types_by_group_id.values())) == 1
        ), node_types_by_group_id  # all groups must have the same node types
        scalar_derivers_by_group_id = {
            group_id: [
                make_scalar_deriver(
                    dst=dst,
                    dst_config=dst_config,
                )
                for dst, dst_config in dst_and_config_list
            ]
            for group_id, dst_and_config_list in dst_and_config_list_by_group_id.items()
        }
        return cls(scalar_derivers_by_group_id, exclude_bottom_k_by_group_id)

    @property
    def activation_location_type_and_pass_types(self) -> list[ActivationLocationTypeAndPassType]:
        return list(
            {
                sub_activation_location_type_and_pass_type
                for scalar_deriver_list in self.scalar_derivers_by_group_id.values()
                for scalar_deriver in scalar_deriver_list
                for sub_activation_location_type_and_pass_type in scalar_deriver.get_sub_activation_location_type_and_pass_types()
            }
        )

    @property
    def devices_for_raw_activations(self) -> list[torch.device]:
        return list(
            {
                scalar_deriver.device_for_raw_activations
                for scalar_deriver_list in self.scalar_derivers_by_group_id.values()
                for scalar_deriver in scalar_deriver_list
            }
        )
