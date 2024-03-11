"""
(My understanding [Dan]; may have some names or details wrong):

This file contains tools for running forward-mode and backward-mode auto-differentiation
(gradients and Jacobian vector products) on tensor -> tensor functions, and generating scalar
derivers using those derivatives.

Child classes of Reconstituter can be created for classes of tensor->tensor functions operating
on residual stream activations, for example recomputing activations of interest from the residual
stream, or recomputing activation and multiplying by the gradient.

Motivation:

Jacobians of functions are slow to work with naively; when operating on vectors of dimension n, they
involve multiplying many nxn matrices together. Backprop through multiple functions is faster than
multiplying the Jacobian of each function, since it's effectively multiplying an nxn matrix by a
length n vector at each step. Jacobian vector products are like backprop, but in the forward
direction. Starting from a vector defined at an early point in the network, you can multiply that
length n vector by the nxn Jacobian at each step of processing, analogous to backprop.

Gradients are useful for computing direct writes from many upstream nodes to a single downstream
node in parallel. Jacobian vector products are useful for computing direct writes to many downstream
nodes in parallel from a single upstream node. The Reconstituter class is meant to make both easier.
"""

import dataclasses
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import torch
from torch.func import jvp as torch_jvp

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    DETACH_LAYER_NORM_SCALE,
    ActivationIndex,
    AttentionTraceType,
    NodeIndex,
    PreOrPostAct,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.locations import (
    ConstantLayerIndexer,
    IdentityLayerIndexer,
    get_activation_index_for_residual_dst,
    get_previous_residual_dst_for_node_type,
    precedes_final_layer,
)
from neuron_explainer.activations.derived_scalars.reconstituted import (
    make_apply_autoencoder,
    make_reconstituted_activation_fn,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DerivedScalarSource,
    DstConfig,
    ScalarDeriver,
    ScalarSource,
)
from neuron_explainer.activations.derived_scalars.utils import detach_and_clone
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    LayerIndex,
    NodeType,
    PassType,
)
from neuron_explainer.models.transformer import Transformer


def compute_gradient_of_scalar_valued_fn_wrt_activations(
    scalar_valued_fn: Callable[[torch.Tensor], torch.Tensor],
    resid: torch.Tensor,
) -> torch.Tensor:
    """
    scalar_valued_fn takes a vector input and returns a scalar. This function evaluates the gradient
    of the scalar_valued_fn with respect to the vector input, at the vector specified by resid.
    """
    scalar_result = scalar_valued_fn(resid)
    assert scalar_result.shape == (), scalar_result.shape
    scalar_result.backward()
    assert resid.grad is not None
    return resid.grad.detach()


def compute_jvp_of_vector_valued_fn_wrt_activations(
    vector_valued_fn: Callable[[torch.Tensor], torch.Tensor],
    resid: torch.Tensor,
    write_vector: torch.Tensor,
) -> torch.Tensor:
    """
    vector_valued_fn takes a vector input and returns a vector. This function evaluates the jacobian
    of the vector_valued_fn with respect to the vector input, at the vector specified by resid, and
    then multiplies the jacobian by write_vector. pytorch's jvp function is used to perform this
    efficiently (without instantiating the full jacobian matrix). This can be considered a "forward
    prop" or the multiplication of a vector by the derivative of a computation graph in the forward
    direction (rather than the more common reverse direction)
    """
    jacobian_vector_product = torch_jvp(
        vector_valued_fn,
        (resid,),
        (write_vector,),
    )[1]

    return jacobian_vector_product


class Reconstituter(ABC):
    """
    This base class has at its core a tensor -> tensor function, reconstitute_activations,
    that computes some set of activations from the residual stream.
    It can compute gradients of that function with respect to the residual stream, using a scalar
    hook to convert the output of reconstitute_activations to a scalar (`reconstitute_gradient`).
    It can also compute Jacobian-vector products (JVPs) of the Jacobian of reconstitute_activations
    with respect to the residual stream, and some write vector (`reconstitute_jvp`).
    It can also be used to generate ScalarDerivers for the original activation, gradient and JVP
    computations (`make_activation_scalar_deriver`, `make_gradient_scalar_deriver`,
    `make_jvp_scalar_deriver`).
    """

    residual_dst: DerivedScalarType
    requires_other_scalar_source: bool
    _input_activation_shape: tuple[int, ...] | None

    def __init__(self) -> None:
        self._input_activation_shape = None

    @abstractmethod
    def reconstitute_activations(
        self,
        resid: torch.Tensor,
        other_arg: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        """Must be implemented by subclasses. This function takes the residual stream as input,
        and returns the (tensor of) activations."""
        pass

    def make_other_scalar_source(self, dst_config: DstConfig) -> ScalarSource:
        """If requires_other_scalar_source is True, this function must be implemented by subclasses.
        Otherwise, it is not used. This generates a ScalarSource for a second scalar used by
        reconstitute_activations."""
        raise NotImplementedError

    def make_residual_scalar_deriver(self, dst_config: DstConfig) -> ScalarDeriver:
        from neuron_explainer.activations.derived_scalars.make_scalar_derivers import (  # lazy to avoid circular import
            make_scalar_deriver,
        )

        assert self.residual_dst.node_type == NodeType.RESIDUAL_STREAM_CHANNEL
        return make_scalar_deriver(
            dst=self.residual_dst,
            dst_config=dst_config,
        )

    def get_residual_activation_index_for_node_index(
        self, node_index: NodeIndex
    ) -> ActivationIndex:
        """
        Given a node index of interest, return the activation index corresponding to the preceding
        residual stream location for that node index. The activation index corresponds to the entire
        residual stream activation tensor for that layer.
        """
        layer_index = node_index.layer_index
        assert layer_index is not None
        return get_activation_index_for_residual_dst(
            dst=self.residual_dst,
            layer_index=layer_index,
        )

    def get_residual_activation_index_for_trace_config(
        self, trace_config: TraceConfig
    ) -> ActivationIndex:
        """
        Given a node index of interest, return the activation index corresponding to the preceding
        residual stream location for that node index. The activation index corresponds to the entire
        residual stream activation tensor for that layer.
        """
        layer_index = trace_config.layer_index
        assert layer_index is not None
        return get_activation_index_for_residual_dst(
            dst=self.residual_dst,
            layer_index=layer_index,
        )

    def _check_other_arg(self, other_arg: torch.Tensor | None) -> None:
        if self.requires_other_scalar_source:
            assert other_arg is not None
        else:
            assert other_arg is None

    def vector_reshape_hook(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Some activations are >1-d tensors per token, but the torch JVP function expects
        1-d tensor -> 1-d tensor functions as input. This reshapes the input tensor to be 1-d,
        and stores the shape for inverting the reshape later.
        """
        self._input_activation_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def vector_unreshape_hook(self, output_vector: torch.Tensor) -> torch.Tensor:
        assert self._input_activation_shape is not None, "must call vector_reshape_hook first"
        reshaped = output_vector.reshape(self._input_activation_shape)
        self._input_activation_shape = None
        return reshaped

    def reconstitute_gradient(
        self,
        resid: torch.Tensor,
        other_arg: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
        scalar_hook: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD

        self._check_other_arg(other_arg)
        resid = detach_and_clone(resid, requires_grad=True)
        if other_arg is not None:
            other_arg = detach_and_clone(other_arg, requires_grad=False)

        def reconstitute_scalar_activation(resid: torch.Tensor) -> torch.Tensor:
            return scalar_hook(
                self.reconstitute_activations(
                    resid,
                    other_arg,
                    layer_index=layer_index,
                    pass_type=pass_type,
                )
            )

        return compute_gradient_of_scalar_valued_fn_wrt_activations(
            scalar_valued_fn=reconstitute_scalar_activation,
            resid=resid,
        )

    def reconstitute_jvp(
        self,
        resid: torch.Tensor,
        other_arg: torch.Tensor | None,
        write_vector: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD

        self._check_other_arg(other_arg)

        resid = detach_and_clone(resid, requires_grad=True)
        if other_arg is not None:
            other_arg = detach_and_clone(other_arg, requires_grad=False)

        def reconstitute_vector_activation(resid: torch.Tensor) -> torch.Tensor:
            return self.vector_reshape_hook(
                self.reconstitute_activations(
                    resid,
                    other_arg,
                    layer_index=layer_index,
                    pass_type=pass_type,
                )
            )

        return self.vector_unreshape_hook(
            compute_jvp_of_vector_valued_fn_wrt_activations(
                vector_valued_fn=reconstitute_vector_activation,
                resid=resid,
                write_vector=write_vector,
            )
        )

    def make_activation_scalar_deriver(
        self, dst_config: DstConfig, output_dst: DerivedScalarType
    ) -> ScalarDeriver:
        residual_scalar_deriver = self.make_residual_scalar_deriver(dst_config)

        if self.requires_other_scalar_source:

            def reconstitute_activations_2_arg(
                resid: torch.Tensor,
                other: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:
                return self.reconstitute_activations(
                    resid,
                    other,
                    layer_index=layer_index,
                    pass_type=pass_type,
                )

            other_scalar_source = self.make_other_scalar_source(dst_config)

            return residual_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
                layerwise_transform_fn=reconstitute_activations_2_arg,
                pass_type_to_transform=PassType.FORWARD,
                other_scalar_source=other_scalar_source,
                output_dst=output_dst,
            )

        else:

            def reconstitute_activations(
                resid: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:
                return self.reconstitute_activations(
                    resid,
                    None,
                    layer_index=layer_index,
                    pass_type=pass_type,
                )

            return residual_scalar_deriver.apply_layerwise_transform_fn_to_output(
                layerwise_transform_fn=reconstitute_activations,
                pass_type_to_transform=PassType.FORWARD,
                output_dst=output_dst,
            )

    def make_gradient_scalar_deriver(
        self,
        scalar_hook: Callable[[torch.Tensor], torch.Tensor],
        dst_config: DstConfig,
        output_dst: (
            DerivedScalarType | None
        ) = None,  # sometimes, we are OK with not defining a new DST for gradient
        # directions, as they are often not used as standalone scalar derivers
    ) -> ScalarDeriver:
        residual_scalar_deriver = self.make_residual_scalar_deriver(dst_config)
        output_dst = output_dst or residual_scalar_deriver.dst
        assert output_dst is not None
        assert output_dst.node_type == NodeType.RESIDUAL_STREAM_CHANNEL

        if self.requires_other_scalar_source:

            def reconstitute_gradient_2_arg(
                resid: torch.Tensor,
                other: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:
                return self.reconstitute_gradient(
                    resid,
                    other,
                    layer_index=layer_index,
                    pass_type=pass_type,
                    scalar_hook=scalar_hook,
                )

            other_scalar_source = self.make_other_scalar_source(dst_config)

            return residual_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
                layerwise_transform_fn=reconstitute_gradient_2_arg,
                pass_type_to_transform=PassType.FORWARD,
                other_scalar_source=other_scalar_source,
                output_dst=output_dst,
            )

        else:

            def reconstitute_gradient(
                resid: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:
                return self.reconstitute_gradient(
                    resid,
                    None,
                    layer_index=layer_index,
                    pass_type=pass_type,
                    scalar_hook=scalar_hook,
                )

            return residual_scalar_deriver.apply_layerwise_transform_fn_to_output(
                layerwise_transform_fn=reconstitute_gradient,
                pass_type_to_transform=PassType.FORWARD,
                output_dst=output_dst,
            )

    def make_jvp_scalar_deriver(
        self,
        write_scalar_source: ScalarSource,
        dst_config: DstConfig,
        output_dst: DerivedScalarType,
    ) -> ScalarDeriver:
        residual_scalar_deriver = self.make_residual_scalar_deriver(dst_config)

        write_precedes_jacobian_layer = partial(
            precedes_final_layer,
            derived_scalar_location_within_layer=write_scalar_source.location_within_layer,
            derived_scalar_layer_index=write_scalar_source.layer_index,
            final_residual_location_within_layer=residual_scalar_deriver.location_within_layer,
        )

        if self.requires_other_scalar_source:

            def reconstitute_jvp_tuple_arg(
                activation_data_tuple: tuple[torch.Tensor, ...],
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:
                resid, other_arg, write_vector = activation_data_tuple
                jacobian_vector_product = self.reconstitute_jvp(
                    resid,
                    other_arg,
                    write_vector=write_vector,
                    layer_index=layer_index,
                    pass_type=pass_type,
                )
                if write_precedes_jacobian_layer(final_residual_layer_index=layer_index):
                    return jacobian_vector_product
                else:
                    # wasteful, but we require the shape to be correct
                    return torch.zeros_like(jacobian_vector_product)

            resid_scalar_source = DerivedScalarSource(
                scalar_deriver=residual_scalar_deriver,
                pass_type=PassType.FORWARD,
                layer_indexer=IdentityLayerIndexer(),
            )

            other_scalar_source = self.make_other_scalar_source(dst_config)

            return ScalarDeriver(
                dst=output_dst,
                dst_config=dst_config,
                tensor_calculate_derived_scalar_fn=reconstitute_jvp_tuple_arg,
                sub_scalar_sources=(resid_scalar_source, other_scalar_source, write_scalar_source),
            )

        else:

            def reconstitute_jvp(
                resid: torch.Tensor,
                write_vector: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:
                jacobian_vector_product = self.reconstitute_jvp(
                    resid,
                    None,
                    write_vector=write_vector,
                    layer_index=layer_index,
                    pass_type=pass_type,
                )
                if write_precedes_jacobian_layer(final_residual_layer_index=layer_index):
                    return jacobian_vector_product
                else:
                    # wasteful, but we require the shape to be correct
                    return torch.zeros_like(jacobian_vector_product)

            return residual_scalar_deriver.apply_layerwise_transform_fn_to_output_and_other_tensor(
                layerwise_transform_fn=reconstitute_jvp,
                pass_type_to_transform=PassType.FORWARD,
                output_dst=output_dst,
                other_scalar_source=write_scalar_source,
            )


class ActivationReconstituter(Reconstituter):
    """Reconstitute MLP, autoencoder, or attention post-softmax activations."""

    requires_other_scalar_source = False

    def __init__(
        self,
        transformer: Transformer,
        autoencoder_context: AutoencoderContext | None,
        node_type: NodeType,
        pre_or_post_act: PreOrPostAct,
        detach_layer_norm_scale: bool,
        attention_trace_type: AttentionTraceType | None = None,
    ):
        super().__init__()
        self._reconstitute_activations_fn = make_reconstituted_activation_fn(
            transformer=transformer,
            autoencoder_context=autoencoder_context,
            node_type=node_type,
            pre_or_post_act=pre_or_post_act,
            detach_layer_norm_scale=detach_layer_norm_scale,
            attention_trace_type=attention_trace_type,
        )
        self._node_type = node_type
        self._pre_or_post_act = pre_or_post_act
        self._attention_trace_type = attention_trace_type
        self.residual_dst = get_previous_residual_dst_for_node_type(
            node_type=node_type,
            autoencoder_dst=autoencoder_context.dst if autoencoder_context is not None else None,
        )

    @classmethod
    def from_trace_config(
        cls,
        transformer: Transformer,
        autoencoder_context: AutoencoderContext | None,
        trace_config: TraceConfig,
    ) -> "ActivationReconstituter":
        return cls(
            transformer=transformer,
            autoencoder_context=autoencoder_context,
            node_type=trace_config.node_type,
            pre_or_post_act=trace_config.pre_or_post_act,
            attention_trace_type=trace_config.attention_trace_type,
            detach_layer_norm_scale=trace_config.detach_layer_norm_scale,
        )

    @classmethod
    def from_activation_location_type(
        cls,
        transformer: Transformer,
        autoencoder_context: AutoencoderContext | None,
        activation_location_type: ActivationLocationType,
        q_or_k: ActivationLocationType | None,
    ) -> "ActivationReconstituter":
        match activation_location_type:
            case ActivationLocationType.MLP_PRE_ACT:
                node_type = NodeType.MLP_NEURON
                pre_or_post_act = PreOrPostAct.PRE
            case ActivationLocationType.MLP_POST_ACT:
                node_type = NodeType.MLP_NEURON
                pre_or_post_act = PreOrPostAct.POST
            case ActivationLocationType.ATTN_QK_LOGITS:
                node_type = NodeType.ATTENTION_HEAD
                pre_or_post_act = PreOrPostAct.PRE
            case ActivationLocationType.ATTN_QK_PROBS:
                node_type = NodeType.ATTENTION_HEAD
                pre_or_post_act = PreOrPostAct.POST
            case ActivationLocationType.ONLINE_AUTOENCODER_LATENT:
                node_type = NodeType.AUTOENCODER_LATENT
                pre_or_post_act = PreOrPostAct.POST
            case ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT:
                node_type = NodeType.MLP_AUTOENCODER_LATENT
                pre_or_post_act = PreOrPostAct.POST
            case ActivationLocationType.ONLINE_ATTENTION_AUTOENCODER_LATENT:
                node_type = NodeType.ATTENTION_AUTOENCODER_LATENT
                pre_or_post_act = PreOrPostAct.POST
            case _:
                raise ValueError(
                    f"Unsupported activation_location_type: {activation_location_type}"
                )
        if node_type == NodeType.ATTENTION_HEAD:
            match q_or_k:
                case ActivationLocationType.ATTN_QUERY:
                    attention_trace_type = AttentionTraceType.Q
                case ActivationLocationType.ATTN_KEY:
                    attention_trace_type = AttentionTraceType.K
                case None:
                    attention_trace_type = AttentionTraceType.QK
                case _:
                    raise ValueError(f"Unsupported q_or_k: {q_or_k}")
        else:
            attention_trace_type = None
        return cls(
            transformer=transformer,
            autoencoder_context=autoencoder_context,
            node_type=node_type,
            pre_or_post_act=pre_or_post_act,
            attention_trace_type=attention_trace_type,
            detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE,
        )

    def reconstitute_activations(
        self,
        resid: torch.Tensor,
        other_arg: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD
        assert other_arg is None
        return self._reconstitute_activations_fn(
            resid,
            layer_index,
            pass_type,
        )

    def make_scalar_hook_for_trace_config(
        self,
        trace_config: TraceConfig,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        assert trace_config.node_type == self._node_type
        assert trace_config.pre_or_post_act == self._pre_or_post_act
        assert trace_config.attention_trace_type == self._attention_trace_type
        assert trace_config.pass_type == PassType.FORWARD
        assert trace_config.layer_index is not None
        assert trace_config.ndim == 0

        def get_activation_from_layer_activations(layer_activations: torch.Tensor) -> torch.Tensor:
            assert all(
                isinstance(index, int) for index in trace_config.tensor_indices
            ), "All indices in trace_config.tensor_indices must be integers."
            return layer_activations[trace_config.tensor_indices]  # type: ignore

        return get_activation_from_layer_activations

    def make_gradient_scalar_deriver_for_trace_config(
        self,
        trace_config: TraceConfig,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> ScalarDeriver:
        scalar_hook = self.make_scalar_hook_for_trace_config(trace_config)
        assert trace_config.layer_index is not None
        dst_config_for_layer = dataclasses.replace(
            dst_config,
            layer_indices=[trace_config.layer_index],
        )
        return self.make_gradient_scalar_deriver(
            scalar_hook=scalar_hook,
            dst_config=dst_config_for_layer,
            output_dst=output_dst,
        )

    def make_gradient_scalar_source_for_trace_config(
        self,
        trace_config: TraceConfig,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> DerivedScalarSource:
        assert trace_config.layer_index is not None
        gradient_scalar_deriver = self.make_gradient_scalar_deriver_for_trace_config(
            trace_config=trace_config,
            dst_config=dst_config,
            output_dst=output_dst,
        )
        return DerivedScalarSource(
            scalar_deriver=gradient_scalar_deriver,
            pass_type=PassType.FORWARD,
            layer_indexer=ConstantLayerIndexer(trace_config.layer_index),
        )

    def make_reconstitute_gradient_fn_for_trace_config(
        self,
        trace_config: TraceConfig,
    ) -> Callable[[torch.Tensor, LayerIndex, PassType], torch.Tensor]:
        scalar_hook = self.make_scalar_hook_for_trace_config(trace_config)

        def reconstitute_gradient(
            resid: torch.Tensor, layer_index: LayerIndex, pass_type: PassType
        ) -> torch.Tensor:
            assert pass_type == PassType.FORWARD
            assert layer_index == trace_config.layer_index
            return self.reconstitute_gradient(
                resid,
                None,
                layer_index=layer_index,
                pass_type=pass_type,
                scalar_hook=scalar_hook,
            )

        return reconstitute_gradient

    def make_reconstitute_activation_fn_for_trace_config(
        self,
        trace_config: TraceConfig,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        scalar_hook = self.make_scalar_hook_for_trace_config(trace_config)

        def reconstitute_activation(
            resid: torch.Tensor,
        ) -> torch.Tensor:
            return scalar_hook(
                self.reconstitute_activations(
                    resid=resid,
                    other_arg=None,
                    layer_index=trace_config.layer_index,
                    pass_type=trace_config.pass_type,
                )
            )

        return reconstitute_activation


def make_no_backward_pass_scalar_source_for_final_residual_grad(
    dst_config: DstConfig,
) -> DerivedScalarSource:
    """This ScalarDeriver takes a residual stream as input and reconstitutes the gradient of the
    scalar valued function specified by dst_config.trace_config, with respect to the
    residual stream."""

    trace_config = dst_config.trace_config
    assert trace_config is not None

    transformer = dst_config.get_or_create_model()
    autoencoder_context = dst_config.get_autoencoder_context()
    autoencoder_dst = autoencoder_context.dst if autoencoder_context is not None else None
    assert trace_config.layer_index is not None

    dst_config_for_layer = dataclasses.replace(
        dst_config,
        layer_indices=[trace_config.layer_index],
    )

    reconstituter = ActivationReconstituter.from_trace_config(
        transformer=transformer,
        autoencoder_context=autoencoder_context,
        trace_config=trace_config,
    )

    return reconstituter.make_gradient_scalar_source_for_trace_config(
        trace_config=trace_config,
        dst_config=dst_config_for_layer,
    )


def make_reconstituted_gradient_fn(
    transformer: Transformer,
    autoencoder_context: AutoencoderContext | None,
    trace_config: TraceConfig,
) -> Callable[[torch.Tensor, LayerIndex, PassType], torch.Tensor]:
    """
    Define a function which takes in a 2-d residual stream tensor (along with layer index and pass
    type) and returns the gradient with respect to trace_config at those residual stream values.
    This function is asserted to be applied at the layer index and pass type specified by
    trace_config.
    """

    assert trace_config.layer_index is not None

    reconstituter = ActivationReconstituter.from_trace_config(
        transformer=transformer,
        autoencoder_context=autoencoder_context,
        trace_config=trace_config,
    )

    return reconstituter.make_reconstitute_gradient_fn_for_trace_config(
        trace_config=trace_config,
    )


class WriteLatentReconstituter(Reconstituter):
    """Reconstitute autoencoder latents from RESID_DELTA_ATTN or RESID_DELTA_MLP."""

    requires_other_scalar_source = False

    def __init__(
        self,
        autoencoder_context: AutoencoderContext,
    ):
        super().__init__()
        self._reconstitute_activations_fn = make_apply_autoencoder(
            autoencoder_context=autoencoder_context,
            use_no_grad=False,
        )
        self.residual_dst = autoencoder_context.dst
        assert self.residual_dst in {
            DerivedScalarType.RESID_DELTA_ATTN,
            DerivedScalarType.RESID_DELTA_MLP,
        }
        assert self.residual_dst.node_type == NodeType.RESIDUAL_STREAM_CHANNEL

    def reconstitute_activations(
        self,
        resid: torch.Tensor,
        other_arg: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD
        assert other_arg is None
        return self._reconstitute_activations_fn(
            resid,
            layer_index,
        )

    def make_scalar_hook_for_latent_index(
        self, latent_index: NodeIndex
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        assert latent_index.pass_type == PassType.FORWARD
        assert latent_index.layer_index is not None
        assert latent_index.ndim == 1

        assert latent_index.tensor_indices[0] is None
        assert isinstance(latent_index.tensor_indices[1], int)
        latent_index_for_grad = latent_index.tensor_indices[1]

        def get_activation_from_layer_activations(layer_activations: torch.Tensor) -> torch.Tensor:
            return layer_activations[:, latent_index_for_grad].sum(dim=0)  # sum over tokens

        return get_activation_from_layer_activations

    def make_gradient_scalar_deriver_for_latent_index(
        self,
        latent_index: NodeIndex,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> ScalarDeriver:
        scalar_hook = self.make_scalar_hook_for_latent_index(latent_index)
        return self.make_gradient_scalar_deriver(
            scalar_hook=scalar_hook,
            dst_config=dst_config,
            output_dst=output_dst,
        )

    def make_gradient_scalar_source_for_latent_index(
        self,
        latent_index: NodeIndex,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> DerivedScalarSource:
        gradient_scalar_deriver = self.make_gradient_scalar_deriver_for_latent_index(
            latent_index=latent_index,
            dst_config=dst_config,
            output_dst=output_dst,
        )
        assert latent_index.layer_index is not None
        return DerivedScalarSource(
            scalar_deriver=gradient_scalar_deriver,
            pass_type=PassType.FORWARD,
            layer_indexer=ConstantLayerIndexer(latent_index.layer_index),
        )
