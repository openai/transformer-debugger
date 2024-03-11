from typing import Any

import torch

from neuron_explainer.activations.derived_scalars.indexing import AttentionTraceType, PreOrPostAct
from neuron_explainer.activations.derived_scalars.reconstituter_class import ActivationReconstituter
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.hooks import AtLayers, TransformerHooks
from neuron_explainer.models.model_component_registry import NodeType, PassType
from neuron_explainer.models.model_context import StandardModelContext


async def test_reconstituter_vs_transformer(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    """
    This test compares the values of activations hooked during the transformer forward
    pass, with the same activations reconstituted from the preceding residual stream
    using an ActivationReconstituter object.
    """
    transformer = standard_model_context.get_or_create_model()

    prompt = "This is a test"
    input_token_ints = torch.tensor(
        standard_model_context.encode(prompt), device=standard_model_context.device
    ).unsqueeze(0)

    test_layer_index = 5

    settings_list: list[dict[str, Any]] = [
        {
            "act_location_type": "mlp.post_act",
            "resid_location_type": "resid.torso.post_attn",
            "node_type": NodeType.MLP_NEURON,
            "pre_or_post_act": PreOrPostAct.POST,
        },
        {
            "act_location_type": "mlp.pre_act",
            "resid_location_type": "resid.torso.post_attn",
            "node_type": NodeType.MLP_NEURON,
            "pre_or_post_act": PreOrPostAct.PRE,
        },
        {
            "act_location_type": "attn.qk_probs",
            "resid_location_type": "resid.torso.post_mlp",
            "node_type": NodeType.ATTENTION_HEAD,
            "pre_or_post_act": PreOrPostAct.POST,
        },
        {
            "act_location_type": "attn.qk_logits",
            "resid_location_type": "resid.torso.post_mlp",
            "node_type": NodeType.ATTENTION_HEAD,
            "pre_or_post_act": PreOrPostAct.PRE,
        },
    ]

    for settings in settings_list:
        act_location_type: str = settings["act_location_type"]
        resid_location_type: str = settings["resid_location_type"]
        node_type: NodeType = settings["node_type"]
        pre_or_post_act: PreOrPostAct = settings["pre_or_post_act"]

        act_layer_index = test_layer_index
        if node_type == NodeType.ATTENTION_HEAD:
            # attention is computed from the post-MLP residual stream in the previous layer
            resid_layer_index = act_layer_index - 1
            attention_trace_type = AttentionTraceType.QK  # irrelevant
        else:
            resid_layer_index = act_layer_index
            attention_trace_type = None  # irrelevant

        reconstituter = ActivationReconstituter(
            transformer=transformer,
            autoencoder_context=standard_autoencoder_context,
            node_type=node_type,
            pre_or_post_act=pre_or_post_act,
            detach_layer_norm_scale=True,  # irrelevant
            attention_trace_type=attention_trace_type,
        )

        hooks = TransformerHooks()
        stored_act = {}
        stored_resid = {}

        def act_saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            stored_act["value"] = act[0]  # 0 batch index
            return act  # store the activation value for the latent in question

        def resid_saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            stored_resid["value"] = act[0]  # 0 batch index
            return act  # store the activation value for the latent in question

        hooks = hooks.append_to_path(
            act_location_type + ".fwd",
            AtLayers([act_layer_index]).append(act_saving_hook_fn),
        ).append_to_path(
            resid_location_type + ".fwd",
            AtLayers([resid_layer_index]).append(resid_saving_hook_fn),
        )

        transformer(input_token_ints, hooks=hooks)

        original = stored_act["value"]
        reconstituted = reconstituter.reconstitute_activations(
            resid=stored_resid["value"],
            other_arg=None,
            layer_index=test_layer_index,
            pass_type=PassType.FORWARD,
        )

        assert original.shape == reconstituted.shape

        torch.testing.assert_close(
            original,
            reconstituted,
            msg=f"Failed for {settings['act_location_type']}",
        )
