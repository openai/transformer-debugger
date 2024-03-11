from typing import Any, Callable

import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import AttentionTraceType, PreOrPostAct
from neuron_explainer.models import Autoencoder
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.hooks import AttentionHooks, NormalizationHooks, TransformerHooks
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    LayerIndex,
    NodeType,
    PassType,
)
from neuron_explainer.models.transformer import Norm, Transformer, TransformerLayer

# scalar derivers that take residual stream as input and
# reconstitute activations such as attention post softmax and mlp post activations


def detach_hook(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return x.detach()


def make_hook_getter() -> tuple[Callable[..., Any], Callable[[], Any]]:
    """
    Returns a hook to append, and a function to retrieve the value of the hook.
    The retrieve function must be called after the hook has been called.
    """
    retrieve = {}

    def hook(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        retrieve["value"] = x
        return x

    return hook, lambda: retrieve["value"]


def zero_batch_dim_hook(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """This hook can be applied before unnecessary computations to save compute."""
    return x[:0, ...]


def apply_layer_norm(
    x: torch.Tensor, norm_module: Norm, detach_layer_norm_scale: bool
) -> torch.Tensor:
    hooks = NormalizationHooks()
    if detach_layer_norm_scale:
        hooks = hooks.append_to_path("scale.fwd", detach_hook)
    return norm_module(x, hooks=hooks)


def add_q_k_or_v_detach_hook(
    hooks: AttentionHooks, q_k_or_v: ActivationLocationType | None
) -> None:
    """If q_k_or_v is None, leave everything attached. If q_k_or_v is not None,
    then leave only the corresponding tensor (Q, K, or V) attached."""
    if q_k_or_v is None:
        return

    if q_k_or_v != ActivationLocationType.ATTN_QUERY:
        hooks.q.append_fwd(detach_hook)
    if q_k_or_v != ActivationLocationType.ATTN_KEY:
        hooks.k.append_fwd(detach_hook)
    if q_k_or_v != ActivationLocationType.ATTN_VALUE:
        hooks.v.append_fwd(detach_hook)


def apply_attn_pre_softmax(
    transformer_layer: TransformerLayer,
    q_k_or_v: ActivationLocationType | None,
    resid_post_mlp: torch.Tensor,
    detach_layer_norm_scale: bool,
) -> torch.Tensor:
    attn_input = apply_layer_norm(
        resid_post_mlp.unsqueeze(0),
        transformer_layer.ln_1,
        detach_layer_norm_scale=detach_layer_norm_scale,
    )  # add batch dimension
    hooks = AttentionHooks()
    add_q_k_or_v_detach_hook(hooks, q_k_or_v)
    get_hook, get_attn = make_hook_getter()
    hooks.qk_logits.append_fwd(get_hook)
    # avoid v_out expense
    hooks.v.append_fwd(zero_batch_dim_hook)
    hooks.qk_logits.append_fwd(zero_batch_dim_hook)
    transformer_layer.attn.forward(attn_input, hooks=hooks)
    # remove batch dimension
    return get_attn()[0]


def apply_mlp_act(
    transformer_layer: TransformerLayer,
    resid_post_attn: torch.Tensor,
    detach_layer_norm_scale: bool,
) -> torch.Tensor:
    pre_act = apply_mlp_pre_act(transformer_layer, resid_post_attn, detach_layer_norm_scale)
    post_act = transformer_layer.mlp.act(pre_act)
    return post_act


def apply_mlp_pre_act(
    transformer_layer: TransformerLayer,
    resid_post_attn: torch.Tensor,
    detach_layer_norm_scale: bool,
) -> torch.Tensor:
    post_ln_mlp = apply_layer_norm(
        resid_post_attn.unsqueeze(0),
        transformer_layer.ln_2,
        detach_layer_norm_scale=detach_layer_norm_scale,
    )  # add batch dimension
    pre_act = transformer_layer.mlp.in_layer(post_ln_mlp)
    return pre_act.squeeze(0)  # remove batch dimension


def apply_autoencoder_pre_latent(
    transformer_layer: TransformerLayer,
    autoencoder: Autoencoder,
    resid: torch.Tensor,
    autoencoder_dst: DerivedScalarType,
    detach_layer_norm_scale: bool,
    latent_slice: slice = slice(None),
) -> torch.Tensor:
    """
    Given the residual stream activations preceding an autoencoder to be
    applied to a given DST, first compute the activations of the DST (`to_be_encoded`)
    and then apply the autoencoder to these activations (NOT INCLUDING the autoencoder nonlinearity),
    and return the result.
    """
    match autoencoder_dst:
        case DerivedScalarType.MLP_POST_ACT:
            to_be_encoded = apply_mlp_act(
                transformer_layer,
                resid,
                detach_layer_norm_scale=detach_layer_norm_scale,
            )
        case DerivedScalarType.RESID_DELTA_ATTN:
            to_be_encoded = apply_resid_delta_attn(
                transformer_layer,
                resid,
                detach_layer_norm_scale=detach_layer_norm_scale,
            )
        case DerivedScalarType.RESID_DELTA_MLP:
            to_be_encoded = apply_resid_delta_mlp(
                transformer_layer,
                resid,
                detach_layer_norm_scale=detach_layer_norm_scale,
            )
        case _:
            raise NotImplementedError(autoencoder_dst.node_type)
    return autoencoder.encode_pre_act(to_be_encoded, latent_slice=latent_slice)


def apply_autoencoder_latent(
    transformer_layer: TransformerLayer,
    autoencoder: Autoencoder,
    resid: torch.Tensor,
    autoencoder_dst: DerivedScalarType,
    detach_layer_norm_scale: bool,
) -> torch.Tensor:
    """
    Given the residual stream activations preceding an autoencoder to be
    applied to a given DST, first compute the activations of the DST (`to_be_encoded`)
    and then apply the autoencoder to these activations (INCLUDING the autoencoder nonlinearity),
    and return the result.
    """
    pre_latent = apply_autoencoder_pre_latent(
        transformer_layer,
        autoencoder,
        resid,
        autoencoder_dst,
        detach_layer_norm_scale=detach_layer_norm_scale,
    )
    return autoencoder.activation(pre_latent)


def apply_resid_delta_attn(
    transformer_layer: TransformerLayer, resid_post_mlp: torch.Tensor, detach_layer_norm_scale: bool
) -> torch.Tensor:
    """
    Compute resid_delta_attn (the output of an attention layer) from the residual stream
    just before the layer
    """
    X = resid_post_mlp.unsqueeze(0)
    hooks = TransformerHooks()
    if detach_layer_norm_scale:
        hooks = hooks.append_to_path("resid.torso.ln_attn.scale.fwd", detach_hook)

    # empty hooks and KV cache to match type signature of transformer_layer methods
    # second output is kv_cache, which is not used here
    attn_delta, _ = transformer_layer.attn_block(X, kv_cache=None, pad=None, hooks=hooks)
    return attn_delta.squeeze(0)


def apply_resid_delta_mlp(
    transformer_layer: TransformerLayer,
    resid_post_attn: torch.Tensor,
    detach_layer_norm_scale: bool,
) -> torch.Tensor:
    """
    Compute resid_delta_mlp (the output of an MLP layer) from the residual stream
    just before the layer
    """
    X = resid_post_attn.unsqueeze(0)
    hooks = TransformerHooks()
    if detach_layer_norm_scale:
        hooks = hooks.append_to_path("resid.torso.ln_mlp.scale.fwd", detach_hook)
    # empty hooks to match type signature of transformer_layer methods
    mlp_delta = transformer_layer.mlp_block(X, hooks=hooks)
    return mlp_delta.squeeze(0)


def make_reconstituted_activation_fn(
    transformer: Transformer,
    autoencoder_context: AutoencoderContext | None,
    node_type: NodeType,
    pre_or_post_act: PreOrPostAct | None,
    detach_layer_norm_scale: bool,
    attention_trace_type: AttentionTraceType | None,
) -> Callable[[torch.Tensor, LayerIndex, PassType], torch.Tensor]:
    match node_type:
        case NodeType.ATTENTION_HEAD:
            match attention_trace_type:
                case AttentionTraceType.QK:
                    q_or_k = None
                case AttentionTraceType.Q:
                    q_or_k = ActivationLocationType.ATTN_QUERY
                case AttentionTraceType.K:
                    q_or_k = ActivationLocationType.ATTN_KEY
                case None:
                    raise ValueError(
                        "attention_trace_type must be specified for attention activations"
                    )

            match pre_or_post_act:
                case PreOrPostAct.PRE:

                    def act_fn(
                        resid: torch.Tensor,
                        layer_index: int | None,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        assert pass_type == PassType.FORWARD
                        assert layer_index is not None
                        return apply_attn_pre_softmax(
                            transformer_layer=transformer.xf_layers[layer_index],
                            q_k_or_v=q_or_k,
                            resid_post_mlp=resid,
                            detach_layer_norm_scale=detach_layer_norm_scale,
                        )

                case PreOrPostAct.POST:
                    apply_attn_V_act = make_apply_attn_V_act(
                        transformer=transformer,
                        q_k_or_v=q_or_k,
                        detach_layer_norm_scale=detach_layer_norm_scale,
                    )  # returns attn, V

                    def act_fn(
                        resid: torch.Tensor,
                        layer_index: LayerIndex,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        assert pass_type == PassType.FORWARD
                        assert layer_index is not None
                        return apply_attn_V_act(
                            resid,
                            layer_index,
                            pass_type,
                        )[
                            0
                        ]  # returns attn

                case _:
                    raise NotImplementedError(pre_or_post_act)
        case NodeType.MLP_NEURON:
            match pre_or_post_act:
                case PreOrPostAct.PRE:

                    def act_fn(
                        resid: torch.Tensor,
                        layer_index: int | None,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        assert pass_type == PassType.FORWARD
                        assert layer_index is not None
                        return apply_mlp_pre_act(
                            transformer_layer=transformer.xf_layers[layer_index],
                            resid_post_attn=resid,
                            detach_layer_norm_scale=detach_layer_norm_scale,
                        )

                case PreOrPostAct.POST:

                    def act_fn(
                        resid: torch.Tensor,
                        layer_index: LayerIndex,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        assert pass_type == PassType.FORWARD
                        assert layer_index is not None
                        return apply_mlp_act(
                            transformer_layer=transformer.xf_layers[layer_index],
                            resid_post_attn=resid,
                            detach_layer_norm_scale=detach_layer_norm_scale,
                        )

                case _:
                    raise NotImplementedError(pre_or_post_act)
        case (
            NodeType.AUTOENCODER_LATENT
            | NodeType.MLP_AUTOENCODER_LATENT
            | NodeType.ATTENTION_AUTOENCODER_LATENT
        ):
            assert autoencoder_context is not None
            match pre_or_post_act:
                case PreOrPostAct.PRE:

                    def act_fn(
                        resid: torch.Tensor,
                        layer_index: int | None,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        assert pass_type == PassType.FORWARD
                        assert layer_index is not None
                        return apply_autoencoder_pre_latent(
                            transformer_layer=transformer.xf_layers[layer_index],
                            autoencoder=autoencoder_context.get_autoencoder(layer_index),
                            resid=resid,
                            autoencoder_dst=autoencoder_context.dst,
                            detach_layer_norm_scale=detach_layer_norm_scale,
                        )

                case PreOrPostAct.POST:

                    def act_fn(
                        resid: torch.Tensor,
                        layer_index: LayerIndex,
                        pass_type: PassType,
                    ) -> torch.Tensor:
                        assert pass_type == PassType.FORWARD
                        assert layer_index is not None
                        return apply_autoencoder_latent(
                            transformer_layer=transformer.xf_layers[layer_index],
                            autoencoder=autoencoder_context.get_autoencoder(layer_index),
                            resid=resid,
                            autoencoder_dst=autoencoder_context.dst,
                            detach_layer_norm_scale=detach_layer_norm_scale,
                        )

                case _:
                    raise NotImplementedError(pre_or_post_act)
        case _:
            raise NotImplementedError(node_type)
    return act_fn


def make_apply_attn_V_act(
    transformer: Transformer,
    q_k_or_v: ActivationLocationType | None,
    detach_layer_norm_scale: bool,
) -> Callable[[torch.Tensor, LayerIndex, PassType], tuple[torch.Tensor, torch.Tensor]]:
    """Used in functions that require reconstituting some or all of the attention head
    operation. Supports specifying a stop grad through all but one of Q, K, and V; or
    if q_k_or_v is None, then all of Q, K, and V are backprop'ed through."""

    def apply_attn_V_act(
        resid: torch.Tensor,
        layer_index: LayerIndex,
        pass_type: PassType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert pass_type == PassType.FORWARD
        transformer_layer = transformer.xf_layers[layer_index]

        attn_input = apply_layer_norm(
            resid.unsqueeze(0),
            transformer_layer.ln_1,
            detach_layer_norm_scale=detach_layer_norm_scale,
        )  # add batch dimension

        hooks = AttentionHooks()
        add_q_k_or_v_detach_hook(hooks, q_k_or_v)
        get_hook, get_v = make_hook_getter()
        hooks.v.append_fwd(get_hook)
        get_hook, get_attn = make_hook_getter()
        hooks.qk_probs.append_fwd(get_hook)
        # avoid v_out expense
        hooks.v.append_fwd(zero_batch_dim_hook)
        hooks.qk_probs.append_fwd(zero_batch_dim_hook)
        transformer_layer.attn.forward(attn_input, hooks=hooks)
        # remove batch dimensions
        return get_attn()[0], get_v()[0]

    return apply_attn_V_act


def make_apply_logits(
    transformer: Transformer,
    detach_layer_norm_scale: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply_logits(
        resid_post_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input: (n_sequence_tokens, d_model) residual stream post-mlp activations at final layer.
        Output: (n_sequence_tokens, n_vocab) logprobs for each token in the sequence.
        """
        post_ln_f = apply_layer_norm(
            resid_post_mlp.unsqueeze(0),
            transformer.final_ln,
            detach_layer_norm_scale=detach_layer_norm_scale,
        )  # add batch dimension
        return transformer.unembed(post_ln_f).squeeze(0)  # remove batch dimension

    return apply_logits


def make_apply_logprobs(
    transformer: Transformer,
    detach_layer_norm_scale: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply_logprobs(
        resid_post_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input: (n_sequence_tokens, d_model) residual stream post-mlp activations at final layer.
        Output: (n_sequence_tokens, n_vocab) logprobs for each token in the sequence.
        """
        logits = make_apply_logits(transformer, detach_layer_norm_scale)(resid_post_mlp)
        return torch.log_softmax(logits, dim=-1)

    return apply_logprobs


def make_apply_autoencoder(
    autoencoder_context: AutoencoderContext,
    use_no_grad: bool = True,  # use True to avoid keeping gradient info for autoencoder;
    # TODO: consider deleting in favor of universal non-gradient-keeping at the outside of ScalarDeriver base functions
) -> Callable[[torch.Tensor, LayerIndex], torch.Tensor]:
    """
    Returns a function that takes a tensor of activations and returns a tensor of the autoencoder
    latent representation of each token.
    """
    # TODO(sbills): Resolve the circular import between this file and attention.py.
    from neuron_explainer.activations.derived_scalars.attention import make_reshape_fn

    # reshape activations to be (n_tokens, n_inputs)
    dst = autoencoder_context.dst
    reshape_fn = make_reshape_fn(dst)

    def apply_autoencoder(raw_activations: torch.Tensor, layer_index: LayerIndex) -> torch.Tensor:
        assert (
            layer_index in autoencoder_context.layer_indices
        ), f"Layer index {layer_index} not in {autoencoder_context.layer_indices}"
        autoencoder = autoencoder_context.get_autoencoder(layer_index)
        latent_activations = autoencoder.encode(reshape_fn(raw_activations))
        return latent_activations  # shape (n_tokens, n_latents)

    if use_no_grad:
        apply_autoencoder = torch.no_grad()(apply_autoencoder)

    return apply_autoencoder
