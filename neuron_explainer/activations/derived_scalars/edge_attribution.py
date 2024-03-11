"""
This file contains functions for computing the importance of edges in a transformer computation
graph.

Edges are taken to go from an upstream node (defined to be an MLP neuron, autoencoder latent, or
attention head at a specific token or token pair) to a downstream "subnode" (defined to be an MLP
neuron, autoencoder latent, or attention head Q, K, or V at a specific token or token pair).

Notice that we consider Q, K, V subnodes for attention separately for the downstream partner of the
edge, but lump them together for the upstream partner.

Note that the inputs to an attention head node are specified as either Q, K, or V-mediated, while
the outputs are specified merely as originating from the attention head node.

'Importance' of an edge is defined using act * grad, or 'attribution'.

See eq. 2 of https://arxiv.org/pdf/2310.10348.pdf for more context, but here briefly:

The attribution of an edge can be computed as:
dLoss/d"EdgeActivation" * "EdgeActivation"
 = dLoss/dDownstreamSubNodeActivation * ∂DownstreamSubNodeActivation/∂UpstreamNodeActivation * UpstreamNodeActivation
 = dLoss/dDownstreamSubNodeActivation * dDownstreamSubNodeActivation/dResidualStream * UpstreamNodeWriteToResidualStream
where the Write terms indicate (d_model,) vectors per token or per token pair, dX/dY indicates the
total derivative of X with respect to Y, and ∂X/∂Y indicates the partial derivative. "Total
derivatives" are also known as "gradients", while "partial derivatives" are also known as "direct
writes" to gradient directions (i.e. dLoss/dDownstreamSubNodeActivation is the gradient at the
downstream node, while ∂DownstreamSubNodeActivation/∂UpstreamNodeActivation is the "direct write"
from the upstream to the downstream node). "EdgeActivation" is the considered to be the direct
effect of the upstream node's activation being patched to the downstream subnode's input. The
"Activation" of a node or subnode is considered to be any sufficient statistic for determining that
node's effect on downstream model components (e.g. the activation of a single MLP neuron, or all
d_head channels of an attention query at a particular pair of tokens). "ResidualStream" refers to
the residual stream just before the downstream subnode in question (edges correspond to direct
writes between nodes).

The strategy within this file is:
1. construct a function to compute
[dLoss/dDownstreamSubNodeActivation * dDownstreamSubNodeActivation(ResidualStream)](ResidualStream)
:=DownstreamSubNodeAttribution(ResidualStream)
with a stopgrad on the dLoss/dDownstreamSubNodeActivation term, for ONE OR MORE downstream subnodes.
This is flexible for use with one or more downstream subnodes to support reuse in two contexts: many
upstream to one downstream node, or one upstream to many downstream nodes.
(make_reconstituted_attribution_fn, which is used to construct AttributionReconstituter)
### MANY-UPSTREAM-TO-ONE-DOWNSTREAM CASE ###
2. construct a function to compute dDownstreamSubNodeAttribution/dResidualStream for JUST ONE
downstream subnode (using AttributionReconstituter)
3. construct a ScalarDeriver by taking the inner product of MANY upstream nodes' write vectors with
the gradient of the attribution of ONE downstream subnode
(convert_scalar_deriver_to_out_edge_attribution)
### ONE-UPSTREAM-TO-MANY-DOWNSTREAM CASE ###
4. construct a function to compute dDownstreamSubNodeAttribution/dResidualStream * WriteVector for
MANY downstream subnodes and ONE upstream node (using AttributionReconstituter)
5. construct a ScalarDeriver for the write vector of a single upstream node (note that this write
vector is per token, even if the upstream node is per token pair; in this case it will be the
contribution of just one (sequence token, attended to token) pair to the sequence token)
(make_node_write_scalar_deriver)
6. convert this ScalarDeriver to a ScalarDeriver for the in edge attribution of many downstream
subnodes, originating from one upstream node
(convert_node_write_scalar_deriver_to_in_edge_attribution,
make_in_edge_attribution_scalar_deriver_factory)

An AttributionReconstituter object is used to reconstruct the attribution of the downstream node(s).
The attribution of the edge is computed by taking derivatives of the downstream node attribution
with respect to the residual stream (where derivatives are either gradients, in the case where there
is one downstream node, or Jacobians, in the case where there are many downstream nodes).
"""

import dataclasses
from typing import Callable

import torch

from neuron_explainer.activations.derived_scalars.attention import (
    make_attn_weighted_value_scalar_deriver,
)
from neuron_explainer.activations.derived_scalars.autoencoder import (
    make_online_autoencoder_latent_scalar_deriver_factory,
)
from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.direct_effects import (
    convert_scalar_deriver_to_write_to_direction,
    convert_scalar_deriver_to_write_to_final_residual_grad,
)
from neuron_explainer.activations.derived_scalars.indexing import (
    AttentionTraceType,
    AttnSubNodeIndex,
    NodeIndex,
    PreOrPostAct,
)
from neuron_explainer.activations.derived_scalars.locations import (
    ConstantLayerIndexer,
    IdentityLayerIndexer,
    get_previous_residual_dst_for_node_type,
)
from neuron_explainer.activations.derived_scalars.mlp import get_base_mlp_scalar_deriver
from neuron_explainer.activations.derived_scalars.node_write import (
    make_node_write_scalar_deriver,
    make_node_write_scalar_source,
)
from neuron_explainer.activations.derived_scalars.raw_activations import (
    make_scalar_deriver_factory_for_activation_location_type,
)
from neuron_explainer.activations.derived_scalars.reconstituted import (
    make_apply_attn_V_act,
    make_reconstituted_activation_fn,
)
from neuron_explainer.activations.derived_scalars.reconstituter_class import Reconstituter
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    DerivedScalarSource,
    DstConfig,
    RawScalarSource,
    ScalarDeriver,
    ScalarSource,
)
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    LayerIndex,
    NodeType,
    PassType,
)
from neuron_explainer.models.model_context import StandardModelContext
from neuron_explainer.models.transformer import Transformer


def get_activation_location_type_for_node_type(
    node_type: NodeType, q_k_or_v: ActivationLocationType | None
) -> ActivationLocationType:
    """This returns the activation location associated with a node of a given type, and
    specifying Q, K, or V in the case of attention. This returns an activation location type
    that is sufficient to determine that node's effect on the residual stream (post-softmax
    in the case of Q, K; ATTN_WEIGHTED_SUM_OF_VALUES in the case of V)"""
    match node_type:
        case NodeType.ATTENTION_HEAD:
            assert q_k_or_v is not None
            match q_k_or_v:
                case ActivationLocationType.ATTN_VALUE:
                    return ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES
                case ActivationLocationType.ATTN_QUERY | ActivationLocationType.ATTN_KEY:
                    return ActivationLocationType.ATTN_QK_PROBS
                case _:
                    raise NotImplementedError(q_k_or_v)
        case NodeType.MLP_NEURON:
            assert q_k_or_v is None
            return ActivationLocationType.MLP_POST_ACT
        case NodeType.AUTOENCODER_LATENT:
            assert q_k_or_v is None
            return ActivationLocationType.ONLINE_AUTOENCODER_LATENT
        case NodeType.MLP_AUTOENCODER_LATENT:
            assert q_k_or_v is None
            return ActivationLocationType.ONLINE_MLP_AUTOENCODER_LATENT
        case NodeType.ATTENTION_AUTOENCODER_LATENT:
            assert q_k_or_v is None
            return ActivationLocationType.ONLINE_ATTENTION_AUTOENCODER_LATENT
        case _:
            raise NotImplementedError(node_type)


def make_reconstituted_attribution_fn(
    transformer: Transformer,
    autoencoder_context: AutoencoderContext | None,
    node_type: NodeType,  # the type of the node of interest
    q_k_or_v: (
        ActivationLocationType | None
    ),  # if node_type is ATTENTION_HEAD, this specifies Q, K, or V
    detach_layer_norm_scale: bool,
) -> Callable[[torch.Tensor, torch.Tensor, LayerIndex, PassType], torch.Tensor]:
    """The 'attribution' of a node is taken to be the product of the node's gradient and the node's activation.
    This returns a function to compute the attribution of attention heads (specifically mediated by Q, K, or V), MLP activations,
    or autoencoder activations. The input expected by that function is the residual stream just before the node in question.
    The function returned can be used for further analysis, e.g. computing the gradient of the attribution with respect to the
    input residual stream.

    Note that this can be used to compute the attribution of one OR many downstream subnodes, depending on the tensor_indices_for_grad
    (an empty tensor_indices_for_grad corresponds to the entire layer worth of activations)."""

    assert (q_k_or_v is None) == (
        node_type != NodeType.ATTENTION_HEAD
    )  # for these functions, we require q_k_or_v to be
    # specified if node_type is ATTENTION_HEAD
    match node_type:
        case NodeType.ATTENTION_HEAD:
            match q_k_or_v:
                case ActivationLocationType.ATTN_QUERY | ActivationLocationType.ATTN_KEY:
                    # in all cases but attn value, the attribution fn is the hadamard product of the activation and the gradient
                    # NOTE: q_k_or_v = None covers all non-attention activations
                    match q_k_or_v:
                        case ActivationLocationType.ATTN_QUERY:
                            attention_trace_type = AttentionTraceType.Q
                        case ActivationLocationType.ATTN_KEY:
                            attention_trace_type = AttentionTraceType.K
                        case None:
                            attention_trace_type = AttentionTraceType.QK
                        case _:
                            raise NotImplementedError(q_k_or_v)
                    activation_fn = make_reconstituted_activation_fn(
                        transformer=transformer,
                        autoencoder_context=autoencoder_context,
                        node_type=node_type,
                        pre_or_post_act=PreOrPostAct.POST,
                        detach_layer_norm_scale=detach_layer_norm_scale,
                        attention_trace_type=attention_trace_type,
                    )

                    def attribution_fn(
                        resid: torch.Tensor,
                        grad: torch.Tensor,
                        layer_index: LayerIndex,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        activation = activation_fn(resid, layer_index, pass_type)
                        assert activation.shape == grad.shape, (
                            activation.shape,
                            grad.shape,
                        )
                        return activation * grad

                case ActivationLocationType.ATTN_VALUE:
                    assert (
                        get_activation_location_type_for_node_type(node_type, q_k_or_v)
                        == ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES
                    )
                    apply_attn_V_act = make_apply_attn_V_act(
                        transformer=transformer,
                        q_k_or_v=q_k_or_v,
                        detach_layer_norm_scale=detach_layer_norm_scale,
                    )

                    def attribution_fn(
                        resid: torch.Tensor,
                        grad: torch.Tensor,
                        layer_index: LayerIndex,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        attn, V = apply_attn_V_act(resid, layer_index, pass_type)
                        attn_weighted_V = torch.einsum("qkh,khd->qkhd", attn, V)
                        # grad is w/r/t (attn_weighted_V summed over k, or ATTN_WEIGHTED_SUM_OF_VALUES)
                        return torch.einsum("qkhd,qhd->qkh", attn_weighted_V, grad)

                case _:
                    raise NotImplementedError(q_k_or_v)
        case _:
            activation_fn = make_reconstituted_activation_fn(
                transformer=transformer,
                autoencoder_context=autoencoder_context,
                node_type=node_type,
                pre_or_post_act=PreOrPostAct.POST,
                detach_layer_norm_scale=detach_layer_norm_scale,
                attention_trace_type=None,
            )

            def attribution_fn(
                resid: torch.Tensor,
                grad: torch.Tensor,
                layer_index: LayerIndex,
                pass_type: PassType,
            ) -> torch.Tensor:
                activation = activation_fn(resid, layer_index, pass_type)
                assert activation.shape == grad.shape, (
                    activation.shape,
                    grad.shape,
                )
                return activation * grad

    return attribution_fn


class AttributionReconstituter(Reconstituter):
    """Reconstitute MLP, autoencoder, or attention node attribution (act * grad at node).
    Attention nodes are required to be split into Q, K, or V subnodes."""

    requires_other_scalar_source = True

    def __init__(
        self,
        transformer: Transformer,
        autoencoder_context: AutoencoderContext | None,
        node_type: NodeType,
        q_k_or_v: ActivationLocationType | None,
        detach_layer_norm_scale: bool,
    ):
        super().__init__()
        self._reconstitute_activations_fn = make_reconstituted_attribution_fn(
            transformer=transformer,
            autoencoder_context=autoencoder_context,
            node_type=node_type,
            q_k_or_v=q_k_or_v,
            detach_layer_norm_scale=detach_layer_norm_scale,
        )
        self.node_type = node_type
        self.q_k_or_v = q_k_or_v
        self.residual_dst = get_previous_residual_dst_for_node_type(
            node_type=node_type,
            autoencoder_dst=autoencoder_context.dst if autoencoder_context is not None else None,
        )

    def reconstitute_activations(
        self,
        resid: torch.Tensor,
        grad: torch.Tensor | None,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> torch.Tensor:
        assert pass_type == PassType.FORWARD
        assert grad is not None
        return self._reconstitute_activations_fn(
            resid,
            grad,
            layer_index,
            pass_type,
        )

    def make_other_scalar_source(self, _unused_dst_config: DstConfig) -> ScalarSource:
        activation_location_type = get_activation_location_type_for_node_type(
            node_type=self.node_type,
            q_k_or_v=self.q_k_or_v,
        )
        return RawScalarSource(
            activation_location_type=activation_location_type,
            pass_type=PassType.BACKWARD,
            layer_indexer=IdentityLayerIndexer(),
        )  # this provides the 'grad' argument required by reconstitute_activations

    def _check_node_index(self, node_index: NodeIndex) -> None:
        assert node_index.node_type == self.node_type
        assert node_index.pass_type == PassType.FORWARD
        assert node_index.layer_index is not None
        if node_index.node_type == NodeType.ATTENTION_HEAD:
            assert isinstance(node_index, AttnSubNodeIndex)
            assert node_index.q_k_or_v == self.q_k_or_v

    def make_scalar_hook_for_node_index(
        self, node_index: NodeIndex
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        self._check_node_index(node_index)

        def get_activation_from_layer_activations(layer_activations: torch.Tensor) -> torch.Tensor:
            return layer_activations[node_index.tensor_indices]

        return get_activation_from_layer_activations

    def make_gradient_scalar_deriver_for_node_index(
        self,
        node_index: NodeIndex,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> ScalarDeriver:
        self._check_node_index(node_index)
        assert node_index.layer_index is not None
        dst_config_for_layer = dataclasses.replace(
            dst_config,
            layer_indices=[node_index.layer_index],
        )
        scalar_hook = self.make_scalar_hook_for_node_index(node_index)
        return self.make_gradient_scalar_deriver(
            scalar_hook=scalar_hook,
            dst_config=dst_config_for_layer,
            output_dst=output_dst,
        )

    def make_gradient_scalar_source_for_node_index(
        self,
        node_index: NodeIndex,
        dst_config: DstConfig,
        output_dst: DerivedScalarType | None = None,
    ) -> DerivedScalarSource:
        scalar_hook = self.make_scalar_hook_for_node_index(node_index)
        gradient_scalar_deriver = self.make_gradient_scalar_deriver(
            scalar_hook=scalar_hook,
            dst_config=dst_config,
            output_dst=output_dst,
        )
        assert node_index.layer_index is not None
        return DerivedScalarSource(
            scalar_deriver=gradient_scalar_deriver,
            pass_type=PassType.FORWARD,
            layer_indexer=ConstantLayerIndexer(node_index.layer_index),
        )


def _make_attribution_reconstituter_for_one_downstream_node(
    dst_config: DstConfig,
) -> AttributionReconstituter:
    # in the case of computing attribution of edges from many upstream to one downstream node, the dst_config
    # contains the information necessary to construct the Reconstituter. This is because the activation being
    # reconstituted corresponds to dst_config.node_index_for_attribution
    node_index_for_attribution = dst_config.node_index_for_attribution
    assert node_index_for_attribution is not None
    node_type = node_index_for_attribution.node_type
    if isinstance(node_index_for_attribution, AttnSubNodeIndex):
        q_k_or_v = node_index_for_attribution.q_k_or_v
    else:
        q_k_or_v = None
    assert (node_type == NodeType.ATTENTION_HEAD) == (q_k_or_v is not None)
    model_context = dst_config.get_model_context()
    transformer = model_context.get_or_create_model()
    autoencoder_context = dst_config.get_autoencoder_context()
    return AttributionReconstituter(
        transformer=transformer,
        autoencoder_context=autoencoder_context,
        node_type=node_type,
        q_k_or_v=q_k_or_v,
        detach_layer_norm_scale=dst_config.detach_layer_norm_scale_for_attribution,
    )


### MANY-UPSTREAM-TO-ONE-DOWNSTREAM CASE ###


def make_grad_of_downstream_subnode_attribution_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Computes the gradient with respect to the preceding residual stream of the downstream subnode's attribution
    (d(Activation(ResidualStream) * Gradient)/dResidualStream), with a stopgrad on the "Gradient" term.
    """
    node_index = dst_config.node_index_for_attribution
    assert node_index is not None
    reconstituter = _make_attribution_reconstituter_for_one_downstream_node(dst_config)
    return reconstituter.make_gradient_scalar_deriver_for_node_index(
        node_index=node_index,
        dst_config=dst_config,
        output_dst=DerivedScalarType.GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION,
    )


def convert_scalar_deriver_to_out_edge_attributions(
    scalar_deriver: ScalarDeriver,
    output_dst: DerivedScalarType,
) -> ScalarDeriver:
    """Converts a scalar deriver for an activation of some kind to a scalar deriver for the
    attribution of edges going out from that activation to the node specified by
    trace_config (which can be autoencoder, MLP, or attention head-- and in the case of
    attention head, specifically the edge going to Q, K, or V)."""

    reconstituter = _make_attribution_reconstituter_for_one_downstream_node(
        scalar_deriver.dst_config,
    )
    node_index = scalar_deriver.dst_config.node_index_for_attribution
    assert node_index is not None
    attribution_grad_scalar_source = reconstituter.make_gradient_scalar_source_for_node_index(
        node_index=node_index,
        dst_config=scalar_deriver.dst_config,
        output_dst=DerivedScalarType.GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION,
    )
    return convert_scalar_deriver_to_write_to_direction(
        scalar_deriver=scalar_deriver,
        direction_scalar_source=attribution_grad_scalar_source,
        output_dst=output_dst,
    )


def make_attn_out_edge_attribution_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the attention value weighted by the post-softmax
    attention between each pair of tokens."""
    attn_weighted_value_scalar_deriver = make_attn_weighted_value_scalar_deriver(dst_config)
    return convert_scalar_deriver_to_out_edge_attributions(
        scalar_deriver=attn_weighted_value_scalar_deriver,
        output_dst=DerivedScalarType.ATTN_OUT_EDGE_ATTRIBUTION,
    )


def make_mlp_out_edge_attribution_scalar_deriver(
    dst_config: DstConfig,
) -> ScalarDeriver:
    """Returns a scalar deriver for the edge attribution of the MLP output layer at each token."""
    scalar_deriver = get_base_mlp_scalar_deriver(
        dst_config=dst_config,
    )
    return convert_scalar_deriver_to_out_edge_attributions(
        scalar_deriver=scalar_deriver,
        output_dst=DerivedScalarType.MLP_OUT_EDGE_ATTRIBUTION,
    )


def make_online_autoencoder_out_edge_attribution_scalar_deriver(
    dst_config: DstConfig,
    node_type: NodeType | None = None,
) -> ScalarDeriver:
    """Returns a scalar deriver for the edge attribution of the MLP output layer at each token."""
    scalar_deriver = make_online_autoencoder_latent_scalar_deriver_factory(node_type)(dst_config)
    return convert_scalar_deriver_to_out_edge_attributions(
        scalar_deriver=scalar_deriver,
        output_dst=DerivedScalarType.ONLINE_AUTOENCODER_OUT_EDGE_ATTRIBUTION,
    )


def make_token_out_edge_attribution_scalar_deriver(dst_config: DstConfig) -> ScalarDeriver:
    """This computes an attribution value for the edge from each token in the sequence to a particular
    downstream node."""

    node_index = dst_config.node_index_for_attribution
    assert node_index is not None
    reconstituter = _make_attribution_reconstituter_for_one_downstream_node(dst_config)
    emb_scalar_deriver = make_scalar_deriver_factory_for_activation_location_type(
        activation_location_type=ActivationLocationType.RESID_POST_EMBEDDING,
    )(dst_config)
    attribution_grad_scalar_source = reconstituter.make_gradient_scalar_source_for_node_index(
        node_index=node_index,
        dst_config=emb_scalar_deriver.dst_config,
        output_dst=DerivedScalarType.GRAD_OF_SINGLE_SUBNODE_ATTRIBUTION,
    )
    return convert_scalar_deriver_to_write_to_direction(
        scalar_deriver=emb_scalar_deriver,
        direction_scalar_source=attribution_grad_scalar_source,
        output_dst=DerivedScalarType.TOKEN_OUT_EDGE_ATTRIBUTION,
    )


### ONE-UPSTREAM-TO-MANY-DOWNSTREAM CASE ###


def convert_node_write_scalar_deriver_to_in_edge_attribution(
    node_write_scalar_source: ScalarSource,
    output_dst: DerivedScalarType,
    dst_config: DstConfig,
    downstream_node_type: NodeType,
    downstream_q_k_or_v: ActivationLocationType | None,
) -> ScalarDeriver:
    """Converts a scalar deriver for a write vector from some upstream node type to a scalar deriver for
    in edge attribution for downstream nodes of some type (MLP, autoencoder, or attention head)."""

    model_context = dst_config.get_model_context()
    autoencoder_context = dst_config.get_autoencoder_context()
    assert isinstance(model_context, StandardModelContext)
    transformer = model_context.get_or_create_model()
    reconstituter = AttributionReconstituter(
        transformer=transformer,
        autoencoder_context=autoencoder_context,
        node_type=downstream_node_type,
        q_k_or_v=downstream_q_k_or_v,
        detach_layer_norm_scale=dst_config.detach_layer_norm_scale_for_attribution,
    )
    return reconstituter.make_jvp_scalar_deriver(
        write_scalar_source=node_write_scalar_source,
        dst_config=dst_config,
        output_dst=output_dst,
    )


def make_in_edge_attribution_scalar_deriver_factory(
    node_type_for_attribution: NodeType,
    q_k_or_v_for_attribution: ActivationLocationType | None = None,
) -> Callable[[DstConfig], ScalarDeriver]:
    """Returns a function that creates a scalar deriver for the edge attribution from arbitrary node
    to the specified downstream node type / sub node type (MLP, autoencoder, or attention head Q, K, or V).
    """

    sub_node_type_to_output_dst = {
        (NodeType.MLP_NEURON, None): DerivedScalarType.MLP_IN_EDGE_ATTRIBUTION,
        (
            NodeType.AUTOENCODER_LATENT,
            None,
        ): DerivedScalarType.ONLINE_AUTOENCODER_IN_EDGE_ATTRIBUTION,
        (
            NodeType.ATTENTION_HEAD,
            ActivationLocationType.ATTN_QUERY,
        ): DerivedScalarType.ATTN_QUERY_IN_EDGE_ATTRIBUTION,
        (
            NodeType.ATTENTION_HEAD,
            ActivationLocationType.ATTN_KEY,
        ): DerivedScalarType.ATTN_KEY_IN_EDGE_ATTRIBUTION,
        (
            NodeType.ATTENTION_HEAD,
            ActivationLocationType.ATTN_VALUE,
        ): DerivedScalarType.ATTN_VALUE_IN_EDGE_ATTRIBUTION,
    }

    output_dst = sub_node_type_to_output_dst[(node_type_for_attribution, q_k_or_v_for_attribution)]

    def make_in_edge_attribution_scalar_deriver(dst_config: DstConfig) -> ScalarDeriver:
        node_write_scalar_source = make_node_write_scalar_source(dst_config)
        return convert_node_write_scalar_deriver_to_in_edge_attribution(
            node_write_scalar_source=node_write_scalar_source,
            output_dst=output_dst,
            dst_config=dst_config,
            downstream_node_type=node_type_for_attribution,
            downstream_q_k_or_v=q_k_or_v_for_attribution,
        )

    return make_in_edge_attribution_scalar_deriver


def make_node_write_to_final_residual_grad_scalar_deriver(dst_config: DstConfig) -> ScalarDeriver:
    """Returns a scalar deriver for the write vector from some upstream node type
    (MLP, autoencoder, or attention head) to the final residual grad. This can be used to compute
    the edge attribution of the edge from that node to the loss itself."""

    node_write_scalar_deriver = make_node_write_scalar_deriver(
        dst_config
    )  # TODO: figure out how to thread
    # the correct layer through to the final residual grad scalar source
    return convert_scalar_deriver_to_write_to_final_residual_grad(
        node_write_scalar_deriver,
        output_dst=DerivedScalarType.SINGLE_NODE_WRITE_TO_FINAL_RESIDUAL_GRAD,
        use_existing_backward_pass_for_final_residual_grad=True,
    )
