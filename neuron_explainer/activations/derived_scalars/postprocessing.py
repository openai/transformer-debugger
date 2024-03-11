from abc import ABC, abstractmethod
from typing import Any

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_store import DerivedScalarStore
from neuron_explainer.activations.derived_scalars.indexing import (
    DETACH_LAYER_NORM_SCALE,
    AttnSubNodeIndex,
    DerivedScalarIndex,
    MirroredNodeIndex,
    NodeIndex,
    PreOrPostAct,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.locations import (
    get_previous_residual_dst_for_node_type,
)
from neuron_explainer.activations.derived_scalars.reconstituter_class import (
    make_reconstituted_gradient_fn,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import DerivedScalarType, DstConfig
from neuron_explainer.models.autoencoder_context import (
    AutoencoderContext,
    MultiAutoencoderContext,
    get_decoder_weight,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    NodeType,
    PassType,
    WeightLocationType,
)
from neuron_explainer.models.model_context import (
    ModelContext,
    StandardModelContext,
    get_embedding,
    get_unembedding_with_ln_gain,
)


class DerivedScalarPostprocessor(ABC):
    """
    A parent class for objects that perform postprocessing on specific tensors of derived scalars.
    This postprocessing in general is assumed to require model weights, hence ModelContext. Optionally,
    it might also require autoencoder weights, hence AutoencoderContext.

    The important logic is in the postprocess() function, which takes a ds_index, and a value
    that was produced using ds_store[ds_index] for some presumed ds_store.
    This produces the postprocessed value. Both the value and the metadata in ds_index might be
    required for performing the computation (e.g. the indices might be used to specify what part of a weight
    tensor is required for performing the computation).
    """

    _input_dst_by_node_type: dict[NodeType, DerivedScalarType]

    # TODO: this should really match derived scalar types based on the compatibility of their indexing prefixes, rather
    # than based on their node_types. Possibly this could take the form of: _input_dst_by_dimensions,
    # and check whether a given derived scalar type's dimensions are a prefix of the input derived scalar type's dimensions.
    # this could avoid the need for the _maybe_convert_input_node_type() method.
    def _extract_tensor_for_postprocessing(
        self,
        node_index: NodeIndex | MirroredNodeIndex,
        ds_store: DerivedScalarStore,
    ) -> tuple[DerivedScalarIndex, torch.Tensor, dict[str, Any]]:
        """
        Finds the ds_index (asserted to be unique) in ds_store that is compatible with node_index,
        (using self.convert_node_index_to_ds_index()),
        and returns the ds_index and the corresponding derived_scalars tensor from ds_store, as well as
        any additional kwargs required by self.postprocess_tensor().

        This allows callers to access derived scalars from ds_store without having to check the derived
        scalar types in the store.
        """
        if isinstance(node_index, MirroredNodeIndex):
            node_index = MirroredNodeIndex.to_node_index(node_index)
        assert isinstance(node_index, NodeIndex), f"{node_index=}"
        assert node_index.pass_type in ds_store.pass_types, (
            f"Pass type {node_index.pass_type} not supported by this DerivedScalarStore; "
            f"supported pass types are {ds_store.pass_types}"
        )
        ds_index = self.convert_node_index_to_ds_index(node_index)
        kwargs = self.get_postprocess_tensor_kwargs(node_index, ds_store)
        return ds_index, ds_store[ds_index], kwargs

    @abstractmethod
    def convert_node_index_to_ds_index(self, node_index: NodeIndex) -> DerivedScalarIndex:
        """For a specified node index, return the corresponding ds_index to submit as arguments to postprocess_tensor()"""
        pass

    def get_postprocess_tensor_kwargs(
        self, node_index: NodeIndex, ds_store: DerivedScalarStore
    ) -> dict[str, Any]:
        """
        Returns a dictionary of keyword arguments that should be passed to postprocess_tensor() for the given node index.
        Varies based on child class, and can be empty.
        """
        return {}

    def postprocess(
        self,
        node_index: NodeIndex | MirroredNodeIndex,
        ds_store: DerivedScalarStore,
    ) -> torch.Tensor:
        """
        The primary function of each child class; takes a node index and a DerivedScalarStore assumed to contain
        the DerivedScalarTypeAndPassType compatible with that node_type, and
        returns a postprocessed value. The postprocessing steps in general depend on any fields of the ds_index,
        as well as additional kwargs defined in self.get_postprocess_tensor_kwargs().
        """

        ds_index, derived_scalars, kwargs = self._extract_tensor_for_postprocessing(
            node_index, ds_store
        )

        return self.postprocess_tensor(ds_index, derived_scalars, **kwargs)

    def postprocess_multiple_nodes(
        self,
        node_indices: list[NodeIndex],
        ds_store: DerivedScalarStore,
    ) -> list[torch.Tensor]:
        """
        A default implementation for postprocessing multiple nodes at once, which calls postprocess() for each node.
        This can be overridden for performance reasons if a more efficient implementation is possible for a given
        DerivedScalarPostprocessor.
        """
        return [self.postprocess(node_index, ds_store) for node_index in node_indices]

    @abstractmethod
    def postprocess_tensor(
        self,
        ds_index: DerivedScalarIndex,
        derived_scalars: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        An alternative entry point for postprocessing, which takes a ds_index and a derived_scalars tensor;
        use this if you do not have access to a full DerivedScalarStore.
        """
        ...

    def get_input_dst_and_config_list(
        self,
        requested_dst_and_config_list: list[tuple[DerivedScalarType, DstConfig]],
    ) -> list[tuple[DerivedScalarType, DstConfig]]:
        """
        This matches the nodes reflected in the requested derived scalar types to the nodes supported by the postprocessor,
        and returns a list of derived scalar types and configurations that should be collected into a DerivedScalarStore to
        be passed to postprocess().
        """
        requested_dsts = [dst for dst, _ in requested_dst_and_config_list]
        dst_configs = [dst_config for _, dst_config in requested_dst_and_config_list]
        requested_node_types = [dst.node_type for dst in requested_dsts]
        assert len(requested_node_types) == len(
            set(requested_node_types)
        ), "Requested derived scalar types must have unique node types"

        input_dsts_and_configs = []
        for i, node_type in enumerate(requested_node_types):
            dst = self._input_dst_by_node_type.get(node_type)
            if dst is not None:
                input_dsts_and_configs.append((dst, dst_configs[i]))

        return input_dsts_and_configs + self.get_constitutive_dst_and_config_list()

    def get_constitutive_dst_and_config_list(self) -> list[tuple[DerivedScalarType, DstConfig]]:
        """
        Returns a list of derived scalar types and configurations that should be collected into a DerivedScalarStore to
        be passed to postprocess(), no matter what the requested derived scalar types are. Varies based on the child
        class, and can be empty.
        """
        return []


class ResidualWriteConverter(DerivedScalarPostprocessor):
    """
    Converts activations to a direction in residual stream space, using write tensors. Valid activations
    are MLP_POST_ACT and ATTN_WEIGHTED_VALUE (equal to post-softmax attention * value), and ONLINE_AUTOENCODER_LATENT
    """

    """
    input dsts and node types accepted by this converter match except in the case of attention heads;
    this is because we require more information (i.e. the entire value vector) to compute token space writes

    node_type == NodeType.V_CHANNEL is a piece of metadata saying that the last index of a derived scalar
    corresponds to a single dimension in v-space, or equivalently
    that if you index all but the last index of the derived scalar, you get a vector in the v-space basis
    """

    _input_dst_by_node_type: dict[NodeType, DerivedScalarType] = {
        NodeType.MLP_NEURON: DerivedScalarType.MLP_POST_ACT,
        NodeType.ATTENTION_HEAD: DerivedScalarType.ATTN_WEIGHTED_VALUE,
        NodeType.LAYER: DerivedScalarType.RESID_POST_EMBEDDING,
    }

    def __init__(
        self,
        model_context: ModelContext,
        multi_autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None,
    ):
        self._model_context = model_context
        self._multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
            multi_autoencoder_context
        )

        if self._multi_autoencoder_context is not None:
            if (
                NodeType.MLP_AUTOENCODER_LATENT
                in self._multi_autoencoder_context.autoencoder_context_by_node_type
            ):
                self._input_dst_by_node_type[
                    NodeType.MLP_AUTOENCODER_LATENT
                ] = DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT
            if (
                NodeType.ATTENTION_AUTOENCODER_LATENT
                in self._multi_autoencoder_context.autoencoder_context_by_node_type
            ):
                self._input_dst_by_node_type[
                    NodeType.ATTENTION_AUTOENCODER_LATENT
                ] = DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT
            if self._multi_autoencoder_context.has_single_autoencoder_context:
                self._input_dst_by_node_type[
                    NodeType.AUTOENCODER_LATENT
                ] = DerivedScalarType.ONLINE_AUTOENCODER_LATENT

    def convert_node_index_to_ds_index(self, node_index: NodeIndex) -> DerivedScalarIndex:
        dst_for_write = self._input_dst_by_node_type[node_index.node_type]
        supported_dsts = list(self._input_dst_by_node_type.values())
        assert dst_for_write in supported_dsts, (
            f"Node type {node_index.node_type} not supported by this DerivedScalarStore; "
            f"supported node types are {supported_dsts}"
        )
        if node_index.node_type == NodeType.LAYER:
            # remove the final, singleton dimension, which is not in the converted derived scalar type
            assert len(node_index.tensor_indices) == 2
            assert node_index.tensor_indices[1] == 0
            updated_tensor_indices: tuple[int | None, ...] = node_index.tensor_indices[:-1]
        else:
            updated_tensor_indices = node_index.tensor_indices
        ds_index = DerivedScalarIndex.from_node_index(
            node_index.with_updates(
                node_type=dst_for_write.node_type, tensor_indices=updated_tensor_indices
            ),
            dst_for_write,
        )
        return ds_index

    def _maybe_decode(
        self, ds_index: DerivedScalarIndex, derived_scalars: torch.Tensor
    ) -> torch.Tensor:
        """decodes if ds_index.dst is an autoencoder latent type, and if so, returns the
        derived_scalars decoded by the decoder weight for the corresponding autoencoder. Otherwise, returns the
        derived_scalars unchanged."""
        if ds_index.dst.is_autoencoder_latent:
            assert self._multi_autoencoder_context is not None
            assert derived_scalars.ndim == 0
            autoencoder = self._multi_autoencoder_context.get_autoencoder(
                ds_index.layer_index, node_type=ds_index.dst.node_type
            )
            assert Dimension.AUTOENCODER_LATENTS in ds_index.tensor_index_by_dim
            indices_for_decoder = (
                ds_index.tensor_index_by_dim[Dimension.AUTOENCODER_LATENTS],
                None,
            )
            slices_for_decoder: tuple[slice | int | None, ...] = tuple(
                slice(None) if index is None else index for index in indices_for_decoder
            )
            decoder_weight = get_decoder_weight(autoencoder)[slices_for_decoder]
            assert decoder_weight.ndim == 1
            derived_scalars = derived_scalars * decoder_weight
        return derived_scalars

    def _get_output_weight(self, ds_index: DerivedScalarIndex) -> torch.Tensor:
        if ds_index.dst.is_autoencoder_latent:
            assert self._multi_autoencoder_context is not None
            autoencoder_context = self._multi_autoencoder_context.get_autoencoder_context(
                ds_index.dst.node_type
            )
            assert autoencoder_context is not None
            output_dst = autoencoder_context.dst
        elif ds_index.dst.node_type == NodeType.ATTENTION_HEAD:
            output_dst = DerivedScalarType.ATTN_WEIGHTED_VALUE
        else:
            output_dst = ds_index.dst

        _output_weight_by_dst: dict[DerivedScalarType, WeightLocationType] = {
            DerivedScalarType.MLP_POST_ACT: WeightLocationType.MLP_TO_RESIDUAL,
            DerivedScalarType.ATTN_WEIGHTED_VALUE: WeightLocationType.ATTN_TO_RESIDUAL,
        }

        assert output_dst in _output_weight_by_dst, f"{output_dst} must be in output weight dict"
        output_weight_location_type = _output_weight_by_dst[output_dst]
        weight_shape_spec = output_weight_location_type.shape_spec

        weight_tensor_indices = tuple(
            [ds_index.tensor_index_by_dim.get(dim, None) for dim in weight_shape_spec]
        )
        weight_tensor_slices: tuple[slice | int | None, ...] = tuple(
            [slice(None) if index is None else index for index in weight_tensor_indices]
        )

        return self._model_context.get_weight(output_weight_location_type, ds_index.layer_index)[
            weight_tensor_slices
        ]

    def postprocess_tensor(
        self, ds_index: DerivedScalarIndex, derived_scalars: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        # TODO: rationalize the setup for choosing the raw activations device by getting it from DerivedScalarTypeConfig,
        # rather than permitting it as an argument to ScalarDeriver __init__.
        # TODO: Derived scalar tensors sometimes haven't been detached yet! We work around that
        # by detaching them here, but we should really just make sure they're always detached.

        assert len(kwargs) == 0, f"Unexpected kwargs: {kwargs}"

        derived_scalars = derived_scalars.to(self._model_context.device).detach()

        # input can be either a scalar or a vector. In the case of e.g. attention heads,
        # a vector worth of information is required to reconstruct the write to the residual stream
        assert derived_scalars.ndim in {0, 1}

        # 1. if an autoencoder latent, return the equivalent model activations; otherwise
        # return the derived scalar unchanged
        derived_scalars = self._maybe_decode(ds_index, derived_scalars)

        # 2. find the output dst
        if ds_index.dst.is_autoencoder_latent:
            assert self._multi_autoencoder_context is not None
            autoencoder_context = self._multi_autoencoder_context.get_autoencoder_context(
                ds_index.dst.node_type
            )
            assert autoencoder_context is not None
            output_dst = autoencoder_context.dst
        else:
            output_dst = ds_index.dst

        # 3. convert from model activations to the residual stream write, unless it is already a residual stream write
        if output_dst.node_type == NodeType.RESIDUAL_STREAM_CHANNEL:
            assert derived_scalars.ndim == 1, f"{ds_index=}, {derived_scalars.shape=}"
            return derived_scalars
        else:
            output_weight = self._get_output_weight(ds_index)
            if derived_scalars.ndim == 0:
                assert (
                    output_dst.node_type == NodeType.MLP_NEURON
                ), f"1-d activation expected only for MLP neurons, not {output_dst.node_type}"
                derived_scalars = derived_scalars.unsqueeze(0)
                assert output_weight.ndim == 1, f"{output_weight.shape=}"
                output_weight = output_weight.unsqueeze(0)
            else:
                assert derived_scalars.ndim == 1
                assert output_weight.ndim == 2
            return torch.einsum("a,ad->d", derived_scalars, output_weight)


class TokenWriteConverter(DerivedScalarPostprocessor):
    """
    Converts activations to a direction in token space, using the unembedding matrix. Valid activations
    are MLP_POST_ACT and ATTN_WEIGHTED_VALUE (equal to post-softmax attention * value), and ONLINE_AUTOENCODER_LATENT
    """

    def __init__(
        self,
        model_context: ModelContext,
        multi_autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None = None,
    ):
        self._model_context = model_context
        self._multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
            multi_autoencoder_context
        )
        self._residual_write_converter = ResidualWriteConverter(
            model_context, multi_autoencoder_context
        )
        self._input_dst_by_node_type = self._residual_write_converter._input_dst_by_node_type
        self._unemb_with_ln_gain = get_unembedding_with_ln_gain(self._model_context)

    def convert_node_index_to_ds_index(self, node_index: NodeIndex) -> DerivedScalarIndex:
        return self._residual_write_converter.convert_node_index_to_ds_index(node_index)

    def postprocess_tensor(
        self, ds_index: DerivedScalarIndex, derived_scalars: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        residual_write = self._residual_write_converter.postprocess_tensor(
            ds_index, derived_scalars, **kwargs
        )

        # 3. convert from the residual stream write to the token-space write
        unembedded_output = torch.einsum("d,dv->v", residual_write, self._unemb_with_ln_gain)

        # 4. subtract the mean, since logprobs are invariant to adding a constant to all logits
        mean_subtracted_unembedded_output = unembedded_output - unembedded_output.mean()

        return mean_subtracted_unembedded_output

    def postprocess_multiple_nodes(
        self,
        node_indices: list[NodeIndex],
        ds_store: DerivedScalarStore,
    ) -> list[torch.Tensor]:
        """
        First, postprocesses all the nodes with the residual write converter, then concatenates them
        and applies the unembedding matrix to the concatenated tensor.
        """
        residual_writes = self._residual_write_converter.postprocess_multiple_nodes(
            node_indices, ds_store
        )
        concatenated_residual_writes = torch.stack(residual_writes, dim=0)
        unembedded_output = torch.einsum(
            "nd,dv->nv", concatenated_residual_writes, self._unemb_with_ln_gain
        )
        mean_subtracted_unembedded_output = unembedded_output - unembedded_output.mean(
            dim=1, keepdim=True
        )
        split_unembedded_output = torch.split(mean_subtracted_unembedded_output, 1, dim=0)
        list_of_tensors = [tensor.squeeze(0) for tensor in split_unembedded_output]
        return list_of_tensors


def _get_residual_stream_tensor_indices_for_node(node_index: NodeIndex) -> tuple[int]:
    """For a given node index defining a point from which the gradient will be computed, this identifies the token
    indices at which the gradient immediately before the node will be nonzero. For attention, in order for there to
    be exactly one such token index, the gradient is computed through one of query/key/value, with a stopgrad
    through the others. Depending on which of query/key/value is used, the token index will be either the query token
    index or the key/value token index. For MLP neurons, the token index will be the token index of the neuron.
    """
    # tensor_indices are expected to be tuple[int, ...], even if length 1
    match node_index.node_type:
        case NodeType.ATTENTION_HEAD:
            # in the case of attention head reads, there are several possible ways to interpret the "read" direction
            # - the gradient through just the query (at the query token)
            # - the gradient through just the key (at the key/value token)
            # - the gradient with respect to some function of the attention write, e.g. the attention write norm,
            # through just the value (at the key/value token)
            assert isinstance(node_index, AttnSubNodeIndex)
            assert len(node_index.tensor_indices) == 3
            match node_index.q_k_or_v:
                case ActivationLocationType.ATTN_QUERY:
                    tensor_index = node_index.tensor_indices[1]  # just the query token index
                case ActivationLocationType.ATTN_KEY | ActivationLocationType.ATTN_VALUE:
                    tensor_index = node_index.tensor_indices[0]  # just the key/value token index
                case _:
                    raise ValueError(f"Unexpected q_k_or_v: {node_index.q_k_or_v}")
        case (
            NodeType.MLP_NEURON
            | NodeType.AUTOENCODER_LATENT
            | NodeType.MLP_AUTOENCODER_LATENT
            | NodeType.ATTENTION_AUTOENCODER_LATENT
        ):
            assert len(node_index.tensor_indices) == 2
            tensor_index = node_index.tensor_indices[0]  # just the token index
        case _:
            raise ValueError(f"Node type {node_index.node_type} not supported")
    assert isinstance(tensor_index, int), (tensor_index, type(tensor_index))
    return (tensor_index,)


class ResidualReadConverter(DerivedScalarPostprocessor):
    """
    Converts activations to a gradient direction in residual stream space, by taking functions that recompute those
    activations from the residual stream, and compute a backward pass on them. Valid activations
    are PREVIOUS_LAYER_RESID_POST_MLP and RESID_POST_ATTN (the DSTs corresponding to residual stream locations that
    precede attention heads and MLP neurons, respectively)
    """

    def __init__(
        self,
        model_context: ModelContext,
        multi_autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None = None,
    ):
        assert isinstance(model_context, StandardModelContext)
        self._transformer = model_context.get_or_create_model()
        self._device = model_context.device
        self._multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
            multi_autoencoder_context
        )

        # TODO: support attention heads; this will require specifying q, k or v in the make_reconstituted_gradient_fn
        self._input_dst_by_node_type: dict[NodeType, DerivedScalarType] = {}
        self._input_dst_by_node_type[NodeType.MLP_NEURON] = get_previous_residual_dst_for_node_type(
            NodeType.MLP_NEURON, None
        )
        self._input_dst_by_node_type[
            NodeType.ATTENTION_HEAD
        ] = get_previous_residual_dst_for_node_type(NodeType.ATTENTION_HEAD, None)
        if self._multi_autoencoder_context is not None:
            # add the autoencoders listed in the multi_autoencoder_context, using their node types
            for (
                node_type,
                autoencoder_context,
            ) in self._multi_autoencoder_context.autoencoder_context_by_node_type.items():
                self._input_dst_by_node_type[node_type] = get_previous_residual_dst_for_node_type(
                    node_type, autoencoder_context.dst
                )
            # if there is only one autoencoder context, also add the "default" node type for backwards compatibility
            if self._multi_autoencoder_context.has_single_autoencoder_context:
                autoencoder_context = (
                    self._multi_autoencoder_context.get_single_autoencoder_context()
                )
                self._input_dst_by_node_type[
                    NodeType.AUTOENCODER_LATENT
                ] = get_previous_residual_dst_for_node_type(
                    NodeType.AUTOENCODER_LATENT, autoencoder_context.dst
                )

    def convert_node_index_to_ds_index(self, node_index: NodeIndex) -> DerivedScalarIndex:
        if node_index.node_type == NodeType.ATTENTION_HEAD:
            # see _get_residual_stream_tensor_indices_for_node for more information
            # TODO: finish supporting attention heads
            assert isinstance(node_index, AttnSubNodeIndex), (
                node_index.node_type,
                type(node_index),
            )
            assert node_index.q_k_or_v in {
                ActivationLocationType.ATTN_QUERY,
                ActivationLocationType.ATTN_KEY,
            }

        dst_for_computing_grad = self._input_dst_by_node_type[node_index.node_type]
        supported_dsts = list(self._input_dst_by_node_type.values())
        assert dst_for_computing_grad in supported_dsts, (
            f"Node type {node_index.node_type} not supported by this DerivedScalarStore; "
            f"supported node types are {supported_dsts}"
        )
        updated_tensor_indices = _get_residual_stream_tensor_indices_for_node(node_index)
        # note: derived scalar indices do not have q_k_or_v associated to them, so we remove this field
        updated_node_index = NodeIndex(
            node_type=dst_for_computing_grad.node_type,
            # Remove the activation index; the entire residual stream will be needed for computing
            # the gradient.
            tensor_indices=updated_tensor_indices,
            layer_index=node_index.layer_index,
            pass_type=node_index.pass_type,
        )
        return DerivedScalarIndex.from_node_index(
            updated_node_index,
            dst_for_computing_grad,
        )

    def get_postprocess_tensor_kwargs(
        self, node_index: NodeIndex, _unused_ds_store: DerivedScalarStore
    ) -> dict[str, Any]:
        return {"node_index": node_index}

    def postprocess_tensor(
        self, ds_index: DerivedScalarIndex, derived_scalars: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        # TODO: rationalize the setup for choosing the raw activations device by getting it from DerivedScalarTypeConfig,
        # rather than permitting it as an argument to ScalarDeriver __init__.
        # TODO: Derived scalar tensors sometimes haven't been detached yet! We work around that
        # by detaching them here, but we should really just make sure they're always detached.

        node_index = kwargs.pop("node_index")
        assert len(kwargs) == 0, f"Unexpected kwargs: {kwargs}"

        assert (
            ds_index.pass_type == PassType.FORWARD
        ), "Residual read converter only supports forward pass"

        derived_scalars = derived_scalars.to(self._device).detach()

        # input should be a residual stream write (1-d)
        assert derived_scalars.ndim == 1

        node_index_with_singleton_first_dim = node_index.with_updates(
            tensor_indices=(0,) + node_index.tensor_indices[1:]
        )
        trace_config = TraceConfig(
            node_index=node_index_with_singleton_first_dim,
            pre_or_post_act=PreOrPostAct.PRE,
            detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE,
        )

        # 1. create the function that computes the residual stream gradient
        if trace_config.node_type.is_autoencoder_latent:
            assert self._multi_autoencoder_context is not None
            autoencoder_context = self._multi_autoencoder_context.get_autoencoder_context(
                trace_config.node_type
            )
            assert autoencoder_context is not None
        else:
            autoencoder_context = None
        reconstitute_gradient = make_reconstituted_gradient_fn(
            transformer=self._transformer,
            autoencoder_context=autoencoder_context,
            trace_config=trace_config,
        )

        # 2. apply the function to the residual stream vector to get the residual stream gradient ("read" vector)
        residual_read = reconstitute_gradient(
            derived_scalars[None], ds_index.layer_index, PassType.FORWARD
        )[
            0
        ]  # add and then remove token dimension for compat with reconstitute_gradient

        return residual_read


class TokenReadConverter(DerivedScalarPostprocessor):
    """
    Converts activations to a direction in token space, by computing a gradient as in ResidualReadConverter,
    and projecting it into token space using the embedding matrix. Valid activations are
    PREVIOUS_LAYER_RESID_POST_MLP and RESID_POST_ATTN (the DSTs corresponding to residual stream locations that
    precede attention heads and MLP neurons, respectively)
    """

    def __init__(
        self,
        model_context: ModelContext,
        multi_autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None = None,
    ):
        self._model_context = model_context
        self._multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
            multi_autoencoder_context
        )
        self._residual_read_converter = ResidualReadConverter(
            model_context, multi_autoencoder_context
        )
        self._input_dst_by_node_type = self._residual_read_converter._input_dst_by_node_type
        self._emb = get_embedding(self._model_context)

    def convert_node_index_to_ds_index(self, node_index: NodeIndex) -> DerivedScalarIndex:
        return self._residual_read_converter.convert_node_index_to_ds_index(node_index)

    def get_postprocess_tensor_kwargs(
        self, node_index: NodeIndex, _unused_ds_store: DerivedScalarStore
    ) -> dict[str, Any]:
        return self._residual_read_converter.get_postprocess_tensor_kwargs(
            node_index, _unused_ds_store
        )

    def postprocess_tensor(
        self, ds_index: DerivedScalarIndex, derived_scalars: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        residual_read = self._residual_read_converter.postprocess_tensor(
            ds_index, derived_scalars, **kwargs
        )

        # 3. convert from the residual stream read to the token-space read
        return torch.einsum("d,vd->v", residual_read, self._emb)

    def postprocess_multiple_nodes(
        self,
        node_indices: list[NodeIndex],
        ds_store: DerivedScalarStore,
    ) -> list[torch.Tensor]:
        """
        First, postprocesses all the nodes with the residual read converter, then concatenates them
        and applies the embedding matrix to the concatenated tensor.
        """
        residual_reads = self._residual_read_converter.postprocess_multiple_nodes(
            node_indices, ds_store
        )
        concatenated_residual_reads = torch.stack(residual_reads, dim=0)
        embedded_output = torch.einsum("nd,vd->nv", concatenated_residual_reads, self._emb)
        split_unembedded_output = torch.split(embedded_output, 1, dim=0)
        list_of_tensors = [tensor.squeeze(0) for tensor in split_unembedded_output]
        return list_of_tensors


class TokenPairAttributionConverter(DerivedScalarPostprocessor):
    """
    Converts activations of an attention-write autoencoder, to compute attribution to each token pair.
    """

    _input_dst_by_node_type: dict[NodeType, DerivedScalarType] = {
        NodeType.ATTENTION_AUTOENCODER_LATENT: DerivedScalarType.ATTENTION_AUTOENCODER_LATENT,
    }

    def __init__(
        self,
        model_context: ModelContext,
        multi_autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None,
        num_tokens_attended_to: int,
    ):
        self._model_context = model_context
        self._multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
            multi_autoencoder_context
        )
        self.num_tokens_attended_to = num_tokens_attended_to

    def postprocess(
        self,
        node_index: NodeIndex | MirroredNodeIndex,
        ds_store: DerivedScalarStore,
    ) -> torch.Tensor:
        if node_index.node_type not in self._input_dst_by_node_type:
            raise ValueError(f"Node type {node_index.node_type} not supported")

        elif self._multi_autoencoder_context is not None:
            autoencoder_context = self._multi_autoencoder_context.get_autoencoder_context(
                node_index.node_type
            )
            if autoencoder_context is None:
                raise ValueError(
                    f"No autoencoder context found for node type {node_index.node_type}."
                )
            if autoencoder_context.dst != DerivedScalarType.RESID_DELTA_ATTN:
                raise ValueError(
                    "Autoencoder context found, but derived scalar type is not RESID_DELTA_ATTN."
                )

        # otherwise proceed
        ds_index, derived_scalars, kwargs = self._extract_tensor_for_postprocessing(
            node_index, ds_store
        )
        return self.postprocess_tensor(ds_index, derived_scalars, **kwargs)

    def convert_node_index_to_ds_index(self, node_index: NodeIndex) -> DerivedScalarIndex:
        dst = self._input_dst_by_node_type[node_index.node_type]
        ds_index = DerivedScalarIndex.from_node_index(
            node_index.with_updates(
                node_type=dst.node_type, tensor_indices=node_index.tensor_indices
            ),
            dst,
        )
        return ds_index

    def postprocess_tensor(
        self, ds_index: DerivedScalarIndex, derived_scalars: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        from neuron_explainer.activations.derived_scalars.autoencoder import (
            make_autoencoder_activation_fn_derivative,
            make_autoencoder_pre_act_encoder_derivative,
        )

        attn_write_sum_heads = kwargs.pop("attn_write_sum_heads")
        assert len(kwargs) == 0, f"Unexpected kwargs: {kwargs}"

        layer_index = ds_index.layer_index
        token_index, latent_index = ds_index.tensor_indices
        assert self._multi_autoencoder_context is not None
        autoencoder_context = self._multi_autoencoder_context.get_autoencoder_context(
            NodeType.ATTENTION_AUTOENCODER_LATENT
        )
        assert autoencoder_context is not None
        assert layer_index is not None

        # compute the activation function derivative
        activation_fn_derivative = make_autoencoder_activation_fn_derivative(
            autoencoder_context, layer_index
        )
        latent_activation = derived_scalars  # (,)
        d_latent_d_pre_act = activation_fn_derivative(latent_activation)  # (,)
        if d_latent_d_pre_act == 0:
            raise ValueError("Latent is inactive.")

        # compute the encoder derivative
        pre_act_encoder_derivative = make_autoencoder_pre_act_encoder_derivative(
            autoencoder_context, layer_index, latent_index=latent_index
        )
        n_tokens_attended_to, d_model = attn_write_sum_heads.shape  # already indexed by token_index
        projection = pre_act_encoder_derivative(attn_write_sum_heads)  # (n_tokens_attended_to, 1)
        projection = projection[:, 0]  # (n_tokens_attended_to,)
        direct_write_to_latents = projection * d_latent_d_pre_act  # (n_tokens_attended_to, )

        # make sure the result has one dimension, because we use zero-dimension when the postprocessor
        # is not supported (return torch.tensor(torch.nan))
        assert direct_write_to_latents.ndim >= 1
        return direct_write_to_latents

    def get_constitutive_dst_and_config_list(self) -> list[tuple[DerivedScalarType, DstConfig]]:
        return [
            (
                DerivedScalarType.ATTN_WRITE_SUM_HEADS,
                DstConfig(
                    model_context=self._model_context,
                ),
            )
        ]

    def get_postprocess_tensor_kwargs(
        self, node_index: NodeIndex, ds_store: DerivedScalarStore
    ) -> dict[str, Any]:
        sequence_token_index = node_index.tensor_indices[0]
        layer_index = node_index.layer_index
        attn_write_sum_heads = ds_store[
            DerivedScalarIndex(
                dst=DerivedScalarType.ATTN_WRITE_SUM_HEADS,
                layer_index=layer_index,
                pass_type=PassType.FORWARD,
                tensor_indices=(sequence_token_index, None),
            )
        ]
        return {"attn_write_sum_heads": attn_write_sum_heads}
