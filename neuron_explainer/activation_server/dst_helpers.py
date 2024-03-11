# Small helper functions for working with derived scalars in the context of activation server
# request handling.

import math
from typing import Any, Callable, TypeVar

import torch

from neuron_explainer.activation_server.requests_and_responses import *
from neuron_explainer.activations.derived_scalars.derived_scalar_store import DerivedScalarStore
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    DerivedScalarIndex,
    MirroredNodeIndex,
)
from neuron_explainer.models.model_component_registry import Dimension

T = TypeVar("T")


def _float_tensor_to_list(x: torch.Tensor) -> list[float]:
    return [x if math.isfinite(x) else -999 for x in x.tolist()]


def _torch_to_tensor_nd(x: torch.Tensor) -> TensorND:
    ndim = x.ndim
    if ndim == 0:
        return Tensor0D(value=x.item())
    elif ndim == 1:
        return Tensor1D(value=_float_tensor_to_list(x))
    elif ndim == 2:
        return Tensor2D(value=[_float_tensor_to_list(row) for row in x])
    elif ndim == 3:
        return Tensor3D(value=[[_float_tensor_to_list(row) for row in matrix] for matrix in x])
    else:
        raise NotImplementedError(f"Unknown ndim: {ndim}")


def _get_dims_to_keep(
    dst: DerivedScalarType, keep_dimension_fn: Callable[[Dimension], bool]
) -> list[Dimension]:
    return [dim for dim in dst.shape_spec_per_token_sequence if keep_dimension_fn(dim)]


def _sum_dst(
    ds_store: DerivedScalarStore,
    dst: DerivedScalarType,
    keep_dimension_fn: Callable[[Dimension], bool],
    abs_mode: bool,
) -> torch.Tensor:
    dims_to_keep = _get_dims_to_keep(dst, keep_dimension_fn)
    store_for_dst = ds_store.filter_dsts([dst])
    activations_and_metadata = next(
        iter(store_for_dst.activations_and_metadata_by_dst_and_pass_type.values())
    )
    ndim_before_sum = len(activations_and_metadata.shape)
    if abs_mode:
        sum_for_dst = store_for_dst.sum_abs(dims_to_keep=dims_to_keep)
    else:
        sum_for_dst = store_for_dst.sum(dims_to_keep=dims_to_keep)
    assert len(sum_for_dst.shape) == len(
        dims_to_keep
    ), f"{sum_for_dst.shape=}, {ndim_before_sum=}, {dims_to_keep=}"
    return sum_for_dst


def get_intermediate_sum_by_dst(
    ds_store: DerivedScalarStore,
    keep_dimension_fn: Callable[[Dimension], bool],
    abs_mode: bool = False,
) -> dict[DerivedScalarType, TensorND]:
    dict_of_torch_tensors = {
        dst: _sum_dst(ds_store, dst, keep_dimension_fn, abs_mode=abs_mode) for dst in ds_store.dsts
    }
    return {dst: _torch_to_tensor_nd(x) for dst, x in dict_of_torch_tensors.items()}


def get_ds_index_from_node_index(
    node_index: MirroredNodeIndex,
    dsts: list[DerivedScalarType],
) -> DerivedScalarIndex:
    """
    Converts from a MirroredNodeIndex (more general, e.g. defined by a NodeType such as MLP neurons)
    to a DerivedScalarIndex (more specific, e.g. defined by a DerivedScalarType such as MLP write
    norm) conditional on the given derived scalar types, which are assumed to be unique for each
    NodeType.
    """
    dsts_matching_node_type = [dst for dst in dsts if dst.node_type == node_index.node_type]
    assert len(dsts_matching_node_type) == 1, (
        f"Expected exactly one derived scalar type to have node type {node_index.node_type}, "
        f"but found {dsts_matching_node_type} in {dsts}"
    )
    return DerivedScalarIndex.from_node_index(
        node_index=node_index,
        dst=dsts_matching_node_type[0],
    )


def assert_tensor(tensor: Any) -> torch.Tensor:
    # for mypy
    assert isinstance(tensor, torch.Tensor)
    return tensor
