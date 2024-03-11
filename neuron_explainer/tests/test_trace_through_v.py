from neuron_explainer.activation_server.derived_scalar_computation import (
    get_derived_scalars_for_prompt,
    maybe_construct_loss_fn_for_backward_pass,
)
from neuron_explainer.activation_server.requests_and_responses import LossFnConfig, LossFnName
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.derived_scalar_store import AttentionTraceType
from neuron_explainer.activations.derived_scalars.indexing import (
    NodeIndex,
    PreOrPostAct,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import DstConfig
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.model_component_registry import NodeType, PassType
from neuron_explainer.models.model_context import StandardModelContext

DETACH_LAYER_NORM_SCALE_FOR_TEST = (
    False  # this sets whether to detach layer norm scale when computing these DSTs.
)


def test_trace_through_v(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    prompt = "This is a test"
    loss_fn_for_backward_pass = maybe_construct_loss_fn_for_backward_pass(
        model_context=standard_model_context,
        config=LossFnConfig(
            name=LossFnName.LOGIT_DIFF,
            target_tokens=["."],
            distractor_tokens=["!"],
        ),
    )

    for downstream_trace_config in [
        None,
        TraceConfig(
            node_index=NodeIndex(
                node_type=NodeType.ATTENTION_HEAD,
                layer_index=5,
                pass_type=PassType.FORWARD,
                tensor_indices=(0, 0, 0),
            ),
            pre_or_post_act=PreOrPostAct.POST,
            detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST,
            attention_trace_type=AttentionTraceType.K,
        ),
    ]:
        trace_config = TraceConfig(
            node_index=NodeIndex(
                node_type=NodeType.ATTENTION_HEAD,
                layer_index=3,
                pass_type=PassType.FORWARD,
                tensor_indices=(0, 0, 0),
            ),
            pre_or_post_act=PreOrPostAct.POST,
            detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST,
            attention_trace_type=AttentionTraceType.V,
            downstream_trace_config=downstream_trace_config,
        )
        dst_config = DstConfig(
            model_context=standard_model_context,
            autoencoder_context=standard_autoencoder_context,
            trace_config=trace_config,
        )
        dst_list = [
            DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
        ]
        dst_and_config_list = [(dst, dst_config) for dst in dst_list]
        current_ds_store, _, raw_store = get_derived_scalars_for_prompt(
            model_context=standard_model_context,
            prompt=prompt,
            trace_config=trace_config,
            dst_and_config_list=dst_and_config_list,  # type: ignore
            autoencoder_context=standard_autoencoder_context,
            loss_fn_for_backward_pass=loss_fn_for_backward_pass,
        )
