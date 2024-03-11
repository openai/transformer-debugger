"""
This file contains classes for referring to individual nodes (e.g. attention heads), activations
(e.g. attention post-softmax), or derived scalars (e.g. attention head write norm) from a forward
pass. DerivedScalarIndex can be used to index into a DerivedScalarStore.

These classes have a parallel structure to each other. One node index can be associated with
multiple activation indices and derived scalar indices. Derived scalar indices can be associated
with more types of scalars that aren't instantiated as 'activations' in the forward pass as
implemented.

Mirrored versions of these classes are used to refer to the same objects, but in a way that can be
transmitted via pydantic response and request data types for communication with a server. Changes
applied to mirrored dataclasses must be applied also to their unmirrored versions, and vice versa.
"""

import dataclasses
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Literal, Union

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    LayerIndex,
    NodeType,
    PassType,
)
from neuron_explainer.pydantic import CamelCaseBaseModel, HashableBaseModel, immutable

DETACH_LAYER_NORM_SCALE = (
    True  # this sets default behavior for whether to detach layer norm scale everywhere
    # TODO: if all goes well, have this be hard-coded to True, and remove the plumbing
)


@dataclass(frozen=True)
class DerivedScalarIndex:
    """
    Indexes into a DerivedScalarStore and returns a tensor of activations specified by indices.
    """

    dst: DerivedScalarType
    tensor_indices: tuple[
        int | None, ...
    ]  # the indices of the activation tensor (not including layer_index)
    # elements of indices correspond to the elements of
    # scalar_deriver.shape_of_activation_per_token_spec
    # e.g. MLP activations might have shape (n_tokens, n_neurons).
    # an element of indices is None -> apply slice(None) for that dimension
    layer_index: LayerIndex  # the layer_index of the activation, if applicable
    pass_type: PassType

    @property
    def tensor_index_by_dim(self) -> dict[Dimension, int | None]:
        tensor_indices_list = list(self.tensor_indices)
        assert len(tensor_indices_list) <= len(self.dst.shape_spec_per_token_sequence), (
            f"Too many tensor indices {tensor_indices_list} for "
            f"{self.dst.shape_spec_per_token_sequence=}"
        )
        tensor_indices_list.extend(
            [None] * (len(self.dst.shape_spec_per_token_sequence) - len(self.tensor_indices))
        )
        return dict(zip(self.dst.shape_spec_per_token_sequence, tensor_indices_list))

    @classmethod
    def from_node_index(
        cls,
        node_index: "NodeIndex | MirroredNodeIndex",
        dst: DerivedScalarType,
    ) -> "DerivedScalarIndex":
        # with the extra information of what dst is desired (subject to the constraint
        # that it must share the same node_type), we can convert a NodeIndex to a DerivedScalarIndex
        assert (
            node_index.node_type == dst.node_type
        ), f"Node type does not match with the derived scalar type: {node_index.node_type=}, {dst=}"
        return cls(
            dst=dst,
            layer_index=node_index.layer_index,
            tensor_indices=node_index.tensor_indices,
            pass_type=node_index.pass_type,
        )


@immutable
class MirroredDerivedScalarIndex(HashableBaseModel):
    dst: DerivedScalarType
    tensor_indices: tuple[int | None, ...]
    layer_index: LayerIndex
    pass_type: PassType

    @classmethod
    def from_ds_index(cls, ds_index: DerivedScalarIndex) -> "MirroredDerivedScalarIndex":
        return cls(
            dst=ds_index.dst,
            layer_index=ds_index.layer_index,
            tensor_indices=ds_index.tensor_indices,
            pass_type=ds_index.pass_type,
        )

    def to_ds_index(self) -> DerivedScalarIndex:
        return DerivedScalarIndex(
            dst=self.dst,
            layer_index=self.layer_index,
            tensor_indices=self.tensor_indices,
            pass_type=self.pass_type,
        )


AllOrOneIndex = Union[int, Literal["All"]]
AllOrOneIndices = tuple[AllOrOneIndex, ...]


@dataclass(frozen=True)
class ActivationIndex:
    """
    This is parallel to DerivedScalarIndex, but specifically for ActivationLocationType's, not for more general DerivedScalarType's.
    """

    activation_location_type: ActivationLocationType
    tensor_indices: AllOrOneIndices
    layer_index: LayerIndex
    pass_type: PassType

    @property
    def tensor_index_by_dim(self) -> dict[Dimension, AllOrOneIndex]:
        # copied from DerivedScalarIndex; TODO: ActivationIndex and DerivedScalarIndex inherit from a shared base class,
        # and perhaps likewise with DerivedScalarType and ActivationLocationType?
        tensor_indices_list = list(self.tensor_indices)
        assert len(tensor_indices_list) <= len(
            self.activation_location_type.shape_spec_per_token_sequence
        ), (
            f"Too many tensor indices {tensor_indices_list} for "
            f"{self.activation_location_type.shape_spec_per_token_sequence=}"
        )
        tensor_indices_list.extend(
            ["All"]
            * (
                len(self.activation_location_type.shape_spec_per_token_sequence)
                - len(self.tensor_indices)
            )
        )
        assert len(tensor_indices_list) == len(
            self.activation_location_type.shape_spec_per_token_sequence
        )
        return dict(
            zip(
                self.activation_location_type.shape_spec_per_token_sequence,
                tensor_indices_list,
            )
        )

    @classmethod
    def from_node_index(
        cls,
        node_index: "NodeIndex | MirroredNodeIndex",
        activation_location_type: ActivationLocationType,
    ) -> "ActivationIndex":
        # with the extra information of what activation_location_type is desired (subject to the constraint
        # that it must share the same node_type), we can convert a NodeIndex to an ActivationIndex
        assert (
            node_index.node_type == activation_location_type.node_type
        ), f"Node type does not match with the derived scalar type: {node_index.node_type=}, {activation_location_type=}"
        return cls(
            activation_location_type=activation_location_type,
            layer_index=node_index.layer_index,
            tensor_indices=make_all_or_one_from_tensor_indices(node_index.tensor_indices),
            pass_type=node_index.pass_type,
        )

    @property
    def ndim(self) -> int:
        return compute_indexed_tensor_ndim(
            activation_location_type=self.activation_location_type,
            tensor_indices=self.tensor_indices,
        )

    def with_updates(self, **kwargs: Any) -> "ActivationIndex":
        """Given new values for fields of this ActivationIndex, return a new ActivationIndex instance with those
        fields updated"""
        return dataclasses.replace(self, **kwargs)


def make_all_or_one_from_tensor_indices(tensor_indices: tuple[int | None, ...]) -> AllOrOneIndices:
    return tuple("All" if tensor_index is None else tensor_index for tensor_index in tensor_indices)


def make_tensor_indices_from_all_or_one_indices(
    all_or_one_indices: AllOrOneIndices,
) -> tuple[int | None, ...]:
    return tuple(
        None if all_or_one_index == "All" else all_or_one_index
        for all_or_one_index in all_or_one_indices
    )


def compute_indexed_tensor_ndim(
    activation_location_type: ActivationLocationType,
    tensor_indices: AllOrOneIndices | tuple[int | None, ...],
) -> int:
    """Returns the dimensionality of a tensor of the given ActivationLocationType after being indexed by tensor_indices.
    int dimensions are removed from the resulting tensor."""
    ndim = activation_location_type.ndim_per_token_sequence - len(
        [tensor_index for tensor_index in tensor_indices if tensor_index not in {"All", None}]
    )
    assert ndim >= 0
    return ndim


def make_python_slice_from_tensor_indices(
    tensor_indices: tuple[int | None, ...]
) -> tuple[slice | int, ...]:
    return make_python_slice_from_all_or_one_indices(
        make_all_or_one_from_tensor_indices(tensor_indices)
    )


def make_python_slice_from_all_or_one_indices(
    all_or_one_indices: AllOrOneIndices,
) -> tuple[slice | int, ...]:
    return tuple(
        slice(None) if all_or_one_index == "All" else all_or_one_index
        for all_or_one_index in all_or_one_indices
    )


@immutable
class MirroredActivationIndex(HashableBaseModel):
    activation_location_type: ActivationLocationType
    tensor_indices: AllOrOneIndices
    layer_index: LayerIndex
    pass_type: PassType

    @classmethod
    def from_activation_index(cls, activation_index: ActivationIndex) -> "MirroredActivationIndex":
        return cls(
            activation_location_type=activation_index.activation_location_type,
            layer_index=activation_index.layer_index,
            tensor_indices=activation_index.tensor_indices,
            pass_type=activation_index.pass_type,
        )

    def to_activation_index(self) -> ActivationIndex:
        return ActivationIndex(
            activation_location_type=self.activation_location_type,
            layer_index=self.layer_index,
            tensor_indices=self.tensor_indices,
            pass_type=self.pass_type,
        )


@dataclass(frozen=True)
class NodeIndex:
    """
    This is parallel to DerivedScalarIndex, but refers to the NodeType associated with a
    DerivedScalarType, rather than the DerivedScalarType itself. This is for situations in
    which multiple derived scalars are computed for the same node.
    """

    node_type: NodeType
    tensor_indices: tuple[int | None, ...]
    layer_index: LayerIndex
    pass_type: PassType

    @classmethod
    def from_ds_index(
        cls,
        ds_index: DerivedScalarIndex,
    ) -> "NodeIndex":
        return cls(
            node_type=ds_index.dst.node_type,
            layer_index=ds_index.layer_index,
            tensor_indices=ds_index.tensor_indices,
            pass_type=ds_index.pass_type,
        )

    @classmethod
    def from_activation_index(
        cls,
        activation_index: ActivationIndex,
    ) -> "NodeIndex":
        return cls(
            node_type=activation_index.activation_location_type.node_type,
            layer_index=activation_index.layer_index,
            tensor_indices=make_tensor_indices_from_all_or_one_indices(
                activation_index.tensor_indices
            ),
            pass_type=activation_index.pass_type,
        )

    def with_updates(self, **kwargs: Any) -> "NodeIndex":
        """Given new values for fields of this NodeIndex, return a new NodeIndex instance with those
        fields updated"""
        return dataclasses.replace(self, **kwargs)

    @property
    def ndim(self) -> int:
        match self.node_type:
            case NodeType.ATTENTION_HEAD:
                reference_activation_location_type = ActivationLocationType.ATTN_QK_PROBS
            case NodeType.MLP_NEURON:
                reference_activation_location_type = ActivationLocationType.MLP_POST_ACT
            case NodeType.AUTOENCODER_LATENT:
                reference_activation_location_type = (
                    ActivationLocationType.ONLINE_AUTOENCODER_LATENT
                )
            case NodeType.MLP_AUTOENCODER_LATENT:
                reference_activation_location_type = (
                    ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT
                )
            case NodeType.ATTENTION_AUTOENCODER_LATENT:
                reference_activation_location_type = (
                    ActivationLocationType.ONLINE_ATTENTION_AUTOENCODER_LATENT
                )
            case NodeType.RESIDUAL_STREAM_CHANNEL:
                reference_activation_location_type = ActivationLocationType.RESID_POST_MLP
            case _:
                raise NotImplementedError(f"Node type {self.node_type} not supported")
        return compute_indexed_tensor_ndim(
            activation_location_type=reference_activation_location_type,
            tensor_indices=self.tensor_indices,
        )

    def to_subnode_index(self, q_k_or_v: ActivationLocationType) -> "AttnSubNodeIndex":
        assert (
            self.node_type == NodeType.ATTENTION_HEAD
        ), f"Node type {self.node_type} is not NodeType.ATTENTION_HEAD"
        return AttnSubNodeIndex(
            node_type=self.node_type,
            layer_index=self.layer_index,
            tensor_indices=self.tensor_indices,
            pass_type=self.pass_type,
            q_k_or_v=q_k_or_v,
        )


@immutable
class MirroredNodeIndex(HashableBaseModel):
    """This class mirrors the fields of NodeIndex without default values."""

    node_type: NodeType
    tensor_indices: tuple[int | None, ...]
    layer_index: LayerIndex
    pass_type: PassType

    @classmethod
    def from_node_index(cls, node_index: NodeIndex) -> "MirroredNodeIndex":
        """
        Note that this conversion may lose information, specifically if the if the NodeIndex
        is an instance of a subclass of NodeIndex such as AttnSubNodeIndex.
        """
        return cls(
            node_type=node_index.node_type,
            layer_index=node_index.layer_index,
            tensor_indices=node_index.tensor_indices,
            pass_type=node_index.pass_type,
        )

    def to_node_index(self) -> NodeIndex:
        return NodeIndex(
            node_type=self.node_type,
            layer_index=self.layer_index,
            tensor_indices=self.tensor_indices,
            pass_type=self.pass_type,
        )


@dataclass(frozen=True)
class AttnSubNodeIndex(NodeIndex):
    """A NodeIndex that contains an extra piece of metadata, q_k_or_v,
    which specifies whether the input to an attention head node should
    be restricted to the portion going through the query, key, or value"""

    q_k_or_v: ActivationLocationType

    def __post_init__(self) -> None:
        assert (
            self.node_type == NodeType.ATTENTION_HEAD
        ), f"Node type {self.node_type} is not NodeType.ATTENTION_HEAD"
        assert self.q_k_or_v in {
            ActivationLocationType.ATTN_QUERY,
            ActivationLocationType.ATTN_KEY,
            ActivationLocationType.ATTN_VALUE,
        }


# TODO: consider subsuming this and the above into NodeIndex/ActivationIndex respectively
@dataclass(frozen=True)
class AttnSubActivationIndex(ActivationIndex):
    """An ActivationIndex that contains an extra piece of metadata, q_or_k,
    which specifies whether the input to an attention head node should
    be restricted to the portion going through the query or key"""

    q_or_k: ActivationLocationType

    def __post_init__(self) -> None:
        assert self.activation_location_type.node_type == NodeType.ATTENTION_HEAD
        assert self.q_or_k in {
            ActivationLocationType.ATTN_QUERY,
            ActivationLocationType.ATTN_KEY,
        }


@immutable
class AblationSpec(CamelCaseBaseModel):
    """A specification for performing ablation on a model."""

    index: MirroredActivationIndex
    value: float


@unique
class AttentionTraceType(Enum):
    Q = "Q"
    K = "K"
    QK = "QK"
    """Q times K"""
    V = "V"
    """Allow gradient to flow through value vector; the attention write * gradient with respect to
    some downstream node or the loss provides the scalar which is backpropagated"""


@immutable
class NodeAblation(CamelCaseBaseModel):
    """A specification for tracing an upstream node.

    This data structure is used by the client. The server converts it to an AblationSpec.
    """

    node_index: MirroredNodeIndex
    value: float


class PreOrPostAct(str, Enum):
    """Specifies whether to trace from pre- or post-nonlinearity"""

    PRE = "pre"
    POST = "post"


@dataclass(frozen=True)
class TraceConfig:
    """This specifies a node from which to compute a backward pass, along with whether to trace from
    pre- or post-nonlinearity, which subnodes to flow the gradient through in the case of an attention node,
    and whether to detach the layer norm scale just before the activation (i.e. whether to flow gradients
    through the layer norm scale parameter)."""

    node_index: NodeIndex
    pre_or_post_act: PreOrPostAct
    detach_layer_norm_scale: bool
    attention_trace_type: AttentionTraceType | None = None  # applies only to attention heads
    downstream_trace_config: "TraceConfig | None" = (
        None  # applies only to attention heads with attention_trace_type == AttentionTraceType.V
    )

    def __post_init__(self) -> None:
        if self.node_index.node_type != NodeType.ATTENTION_HEAD:
            assert self.attention_trace_type is None

        if self.attention_trace_type != AttentionTraceType.V:
            # only tracing through V supports a downstream node
            assert self.downstream_trace_config is None
        else:
            if self.downstream_trace_config is not None:
                # repeatedly tracing through V is not allowed; all other types of
                # downstream trace configs are fine
                assert self.downstream_trace_config.attention_trace_type != AttentionTraceType.V
            # cfg is None -> a loss (function of logits) is assumed to be defined

    @property
    def node_type(self) -> NodeType:
        return self.node_index.node_type

    @property
    def tensor_indices(self) -> AllOrOneIndices:
        return make_all_or_one_from_tensor_indices(self.node_index.tensor_indices)

    @property
    def layer_index(self) -> LayerIndex:
        return self.node_index.layer_index

    @property
    def pass_type(self) -> PassType:
        return self.node_index.pass_type

    @property
    def ndim(self) -> int:
        return self.node_index.ndim

    def with_updated_index(
        self,
        **kwargs: Any,
    ) -> "TraceConfig":
        return dataclasses.replace(
            self,
            node_index=self.node_index.with_updates(**kwargs),
        )

    @classmethod
    def from_activation_index(
        cls,
        activation_index: ActivationIndex,
        detach_layer_norm_scale: bool = DETACH_LAYER_NORM_SCALE,
    ) -> "TraceConfig":
        node_index = NodeIndex.from_activation_index(activation_index)
        match activation_index.activation_location_type:
            case ActivationLocationType.MLP_PRE_ACT | ActivationLocationType.ATTN_QK_LOGITS:
                pre_or_post_act = PreOrPostAct.PRE
            case (
                ActivationLocationType.MLP_POST_ACT
                | ActivationLocationType.ATTN_QK_PROBS
                | ActivationLocationType.ONLINE_AUTOENCODER_LATENT
            ):
                pre_or_post_act = PreOrPostAct.POST
            case _:
                raise ValueError(
                    f"ActivationLocationType {activation_index.activation_location_type} not supported"
                )
        match node_index.node_type:
            case NodeType.ATTENTION_HEAD:
                attention_trace_type: AttentionTraceType | None = AttentionTraceType.QK
            case _:
                attention_trace_type = None
        downstream_trace_config = None
        return cls(
            node_index=node_index,
            pre_or_post_act=pre_or_post_act,
            detach_layer_norm_scale=detach_layer_norm_scale,
            attention_trace_type=attention_trace_type,
            downstream_trace_config=downstream_trace_config,
        )


@immutable
class MirroredTraceConfig(HashableBaseModel):
    node_index: MirroredNodeIndex
    pre_or_post_act: PreOrPostAct
    detach_layer_norm_scale: bool
    attention_trace_type: AttentionTraceType | None = None  # applies only to attention heads
    downstream_trace_config: "MirroredTraceConfig | None" = (
        None  # applies only to attention heads with attention_trace_type == AttentionTraceType.V
    )

    def to_trace_config(self) -> TraceConfig:
        downstream_trace_config = (
            self.downstream_trace_config.to_trace_config()
            if self.downstream_trace_config is not None
            else None
        )
        return TraceConfig(
            node_index=self.node_index.to_node_index(),
            pre_or_post_act=self.pre_or_post_act,
            detach_layer_norm_scale=self.detach_layer_norm_scale,
            attention_trace_type=self.attention_trace_type,
            downstream_trace_config=downstream_trace_config,
        )

    @classmethod
    def from_trace_config(cls, trace_config: TraceConfig) -> "MirroredTraceConfig":
        mirrored_downstream_trace_config = (
            cls.from_trace_config(trace_config.downstream_trace_config)
            if trace_config.downstream_trace_config is not None
            else None
        )
        return cls(
            node_index=MirroredNodeIndex.from_node_index(trace_config.node_index),
            pre_or_post_act=trace_config.pre_or_post_act,
            detach_layer_norm_scale=trace_config.detach_layer_norm_scale,
            attention_trace_type=trace_config.attention_trace_type,
            downstream_trace_config=mirrored_downstream_trace_config,
        )


@immutable
class NodeToTrace(CamelCaseBaseModel):
    """A specification for tracing a node.

    This data structure is used by the client. The server converts it to an activation index and
    an ablation spec.

    In the case of tracing through attention value, there can be up to two NodeToTrace
    objects: one upstream and one downstream. First, a gradient is computed with respect to the
    downstream node. Then, the direct effect of the upstream (attention) node on that downstream
    node is computed. Then, the gradient is computed with respect to that direct effect, propagated
    through V
    """

    node_index: MirroredNodeIndex
    attention_trace_type: AttentionTraceType | None
    downstream_trace_config: MirroredTraceConfig | None
