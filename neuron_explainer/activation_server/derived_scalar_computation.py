"""
This file contains code to add hooks relevant to a list of scalar derivers, and to run forward
passes with those hooks to populate a DerivedScalarStore with the value of the scalars in question.
"""

import gc
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch
from fastapi import HTTPException

from neuron_explainer.activation_server.requests_and_responses import (
    GroupId,
    InferenceAndTokenData,
    InferenceData,
    LossFnConfig,
    LossFnName,
)
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.derived_scalar_store import (
    DerivedScalarStore,
    RawActivationStore,
)
from neuron_explainer.activations.derived_scalars.direct_effects import (
    AttentionDirectEffectReconstituter,
)
from neuron_explainer.activations.derived_scalars.indexing import (
    DETACH_LAYER_NORM_SCALE,
    AblationSpec,
    ActivationIndex,
    AttentionTraceType,
    TraceConfig,
    make_python_slice_from_all_or_one_indices,
)
from neuron_explainer.activations.derived_scalars.logprobs import LogitReconstituter
from neuron_explainer.activations.derived_scalars.multi_group import (
    MultiGroupDerivedScalarStore,
    MultiGroupScalarDerivers,
)
from neuron_explainer.activations.derived_scalars.reconstituter_class import ActivationReconstituter
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    ActivationLocationTypeAndPassType,
    DstConfig,
    ScalarDeriver,
)
from neuron_explainer.activations.hook_graph import AutoencoderHookGraph, TransformerHookGraph
from neuron_explainer.models import Transformer
from neuron_explainer.models.autoencoder_context import AutoencoderContext, MultiAutoencoderContext
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    LayerIndex,
    NodeType,
    PassType,
)
from neuron_explainer.models.model_context import (
    InvalidTokenException,
    ModelContext,
    StandardModelContext,
)
from neuron_explainer.models.transformer import prep_input_and_right_pad_for_forward_pass

T = TypeVar("T")

# a nested dict of lists of tuples, where each tuple contains a DerivedScalarType and a
# DerivedScalarTypeConfig (the necessary information to specify a ScalarDeriver). The nested dict is
# keyed first by spec_name and then by group_id, where spec_name is the name associated with a
# ProcessingRequestSpec, and group_id refers to a GroupId enum value (each GroupId referring to a
# set of DSTs).
DstAndConfigsByProcessingStep = dict[str, dict[GroupId, list[tuple[DerivedScalarType, DstConfig]]]]

# a nested dict of lists of ScalarDerivers, parallel to and constructed from
# DstAndConfigsByProcessingStep
ScalarDeriversByProcessingStep = dict[str, dict[GroupId, list[ScalarDeriver]]]

# a nested dict of DerivedScalarStores, parallel to and constructed using
# ScalarDeriversByProcessingStep; also uses RawActivationStore as input (though note that
# RawActivationStore is a single object used for the entire nested dict)
DerivedScalarStoreByProcessingStep = dict[str, dict[GroupId, DerivedScalarStore]]


@dataclass(frozen=True)
class DerivedScalarComputationParams:
    input_token_ints: list[int]
    multi_group_scalar_derivers_by_processing_step: dict[str, MultiGroupScalarDerivers]
    loss_fn_for_backward_pass: Callable[[torch.Tensor], torch.Tensor] | None
    device_for_raw_activations: torch.device
    ablation_specs: list[AblationSpec] | None
    trace_config: TraceConfig | None

    @property
    def prompt_length(self) -> int:
        return len(self.input_token_ints)

    @property
    def activation_location_type_and_pass_types(self) -> list[ActivationLocationTypeAndPassType]:
        return list(
            {
                alt_and_pt
                for mgsd in self.multi_group_scalar_derivers_by_processing_step.values()
                for alt_and_pt in mgsd.activation_location_type_and_pass_types
            }
        )


def construct_logit_diff_loss_fn(
    model_context: ModelContext,
    target_tokens: list[str],
    distractor_tokens: list[str],
    subtract_mean: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        target_tokens_as_ints = model_context.encode_token_str_list(target_tokens)
        distractor_tokens_as_ints = model_context.encode_token_str_list(distractor_tokens)
    except InvalidTokenException as e:
        raise HTTPException(status_code=400, detail=str(e))

    def loss_fn_for_backward_pass(output_logits: torch.Tensor) -> torch.Tensor:
        assert output_logits.ndim == 3
        nbatch, ntoken, nlogit = output_logits.shape
        assert nbatch == 1
        assert len(target_tokens_as_ints) > 0
        target_mean = output_logits[:, -1, target_tokens_as_ints].mean(-1)
        if len(distractor_tokens_as_ints) == 0:
            loss = target_mean.mean()  # average logits for target tokens
            if subtract_mean:
                loss -= output_logits[:, -1, :].mean()
            return loss
        else:
            assert (
                not subtract_mean
            ), "subtract_mean not a meaningful option when distractor_tokens is specified"
            distractor_mean = output_logits[:, -1, distractor_tokens_as_ints].mean(-1)
            return (
                target_mean - distractor_mean
            ).mean()  # difference between average logits for target and distractor tokens

    return loss_fn_for_backward_pass


def construct_probs_loss_fn(
    model_context: ModelContext, target_tokens: list[str]
) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        target_tokens_as_ints = model_context.encode_token_str_list(target_tokens)
    except InvalidTokenException as e:
        raise HTTPException(status_code=400, detail=str(e))

    def loss_fn_for_backward_pass(output_logits: torch.Tensor) -> torch.Tensor:
        assert output_logits.ndim == 3
        output_probs = torch.softmax(output_logits, dim=-1)
        nbatch, ntoken, nlogit = output_probs.shape
        assert nbatch == 1
        assert len(target_tokens_as_ints) > 0
        target_sum = output_probs[:, -1, target_tokens_as_ints].sum(-1)
        return target_sum.mean()  # average summed probs for target tokens

    return loss_fn_for_backward_pass


def construct_zero_loss_fn() -> Callable[[torch.Tensor], torch.Tensor]:
    """This loss function is used for running a backward pass that will be interrupted
    by ablating some desired parameters. Parameters downstream of the ablated parameters
    will have a gradient of 0, and parameters upstream of the ablated parameters will
    in general have a non-zero gradient."""

    def loss_fn_for_backward_pass(output_logits: torch.Tensor) -> torch.Tensor:
        return 0.0 * output_logits.sum()

    return loss_fn_for_backward_pass


def maybe_construct_loss_fn_for_backward_pass(
    model_context: ModelContext, config: LossFnConfig | None
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if config is None:
        return None
    else:
        if config.name == LossFnName.LOGIT_DIFF:
            assert config.target_tokens is not None
            target_tokens = config.target_tokens
            distractor_tokens = config.distractor_tokens or []

            return construct_logit_diff_loss_fn(
                model_context=model_context,
                target_tokens=target_tokens,
                distractor_tokens=distractor_tokens,
                subtract_mean=False,
            )
        elif config.name == LossFnName.LOGIT_MINUS_MEAN:
            assert config.target_tokens is not None
            assert config.distractor_tokens is None
            return construct_logit_diff_loss_fn(
                model_context=model_context,
                target_tokens=config.target_tokens,
                distractor_tokens=[],
                subtract_mean=True,
            )
        elif config.name == LossFnName.PROBS:
            assert config.target_tokens is not None
            assert config.distractor_tokens is None
            target_tokens = config.target_tokens
            return construct_probs_loss_fn(model_context=model_context, target_tokens=target_tokens)
        elif config.name == LossFnName.ZERO:
            return construct_zero_loss_fn()
        else:
            raise NotImplementedError(f"Unknown loss fn name: {config.name}")


ablatable_activation_location_type_by_node_type = {
    NodeType.MLP_NEURON: ActivationLocationType.MLP_POST_ACT,
    NodeType.ATTENTION_HEAD: ActivationLocationType.ATTN_QK_PROBS,
    NodeType.RESIDUAL_STREAM_CHANNEL: ActivationLocationType.RESID_POST_MLP,
    NodeType.AUTOENCODER_LATENT: ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
}


def compute_derived_scalar_groups_for_input_token_ints(
    model_context: StandardModelContext,
    multi_autoencoder_context: MultiAutoencoderContext | None,
    batched_ds_computation_params: list[DerivedScalarComputationParams],
) -> tuple[
    list[dict[str, MultiGroupDerivedScalarStore]], list[InferenceData], list[RawActivationStore]
]:
    """This function runs a batched forward pass on the given batch of input token sequences, with
    hooks added to the transformer to extract the activations needed to compute the scalars in
    multi_group_scalar_derivers for each batch element. It then returns a batch of dicts of
    DerivedScalarStores by group_id containing the relevant derived scalars for each token in
    the input, as well as a batch of InferenceData objects containing tokenized inputs and other metadata,
    and a batch of RawActivationStores, each of which was used to compute the respective
    dict of DerivedScalarStores. These RawActivationStores can be used to compute additional derived scalars
    post-hoc.
    """

    (
        batched_raw_activation_store,
        batched_loss,
        batched_activation_value_for_backward_pass,
        batched_memory_used_before,
        inference_time,
    ) = run_inference_and_populate_raw_store(
        model_context=model_context,
        multi_autoencoder_context=multi_autoencoder_context,
        batched_ds_computation_params=batched_ds_computation_params,
    )

    assert (
        len(batched_raw_activation_store)
        == len(batched_ds_computation_params)
        == len(batched_loss)
        == len(batched_activation_value_for_backward_pass)
        == len(batched_memory_used_before)
    )

    batched_multi_group_ds_store_by_processing_step: list[
        dict[str, MultiGroupDerivedScalarStore]
    ] = []
    (
        batched_multi_group_ds_store_by_processing_step,
        batched_memory_used_after,
    ) = construct_ds_stores_from_raw(
        batched_raw_activation_store,
        batched_ds_computation_params,
    )

    batched_inference_data: list[InferenceData] = []
    for (
        loss,
        activation_value_for_backward_pass,
        memory_used_before,
        memory_used_after,
    ) in zip(
        batched_loss,
        batched_activation_value_for_backward_pass,
        batched_memory_used_before,
        batched_memory_used_after,
    ):
        inference_data = InferenceData(
            inference_time=inference_time,
            loss=loss,
            activation_value_for_backward_pass=activation_value_for_backward_pass,
            memory_used_before=memory_used_before,
            memory_used_after=memory_used_after,
        )
        batched_inference_data.append(inference_data)
    return (
        batched_multi_group_ds_store_by_processing_step,
        batched_inference_data,
        batched_raw_activation_store,
    )


def get_activation_index_and_reconstitute_activation_fn(
    transformer: Transformer,
    multi_autoencoder_context: MultiAutoencoderContext | None,
    trace_config: TraceConfig,
) -> tuple[ActivationIndex, Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]]:
    """
    This function returns the ActivationIndex corresponding to the preceding residual stream
    index implied by the trace_config. It also returns a function taking one tensor, used to recompute
    the activation specified by the trace_config from the residual stream.
    """
    assert trace_config.attention_trace_type != AttentionTraceType.V
    if trace_config.node_type.is_autoencoder_latent:
        assert multi_autoencoder_context is not None
        autoencoder_context = multi_autoencoder_context.get_autoencoder_context(
            trace_config.node_type
        )
        assert autoencoder_context is not None
    else:
        autoencoder_context = None
    act_reconstituter = ActivationReconstituter.from_trace_config(
        transformer=transformer,
        autoencoder_context=autoencoder_context,
        trace_config=trace_config,
    )
    activation_index_for_reconstituter = act_reconstituter.get_residual_activation_index_for_node_index(
        # convert trace_config to node index
        trace_config.node_index
    )

    def reconstitute_activation_fn(
        upstream_resid: torch.Tensor, _unused_downstream_resid: torch.Tensor | None
    ) -> torch.Tensor:
        assert _unused_downstream_resid is None
        reconstitute_activation = (
            act_reconstituter.make_reconstitute_activation_fn_for_trace_config(
                trace_config=trace_config
            )
        )
        return reconstitute_activation(upstream_resid)

    return activation_index_for_reconstituter, reconstitute_activation_fn


def get_activation_indices_and_reconstitute_direct_effect_fn(
    model_context: ModelContext,
    multi_autoencoder_context: MultiAutoencoderContext | None,
    trace_config: TraceConfig,
    loss_fn_for_backward_pass: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[
    ActivationIndex,
    ActivationIndex,
    Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
]:
    """
    For use with AttentionTraceType.V. This function returns the ActivationIndex corresponding to the
    residual stream before the upstream (attention V) node as well as the ActivationIndex corresponding to the
    residual stream before the downstream node, or before the loss. It also returns a function taking two arguments,
    both residual stream activations, which computes the direct effect of the upstream activation on the downstream
    activation.
    If trace_config.attention_trace_config.downstream_trace_config is a normal TraceConfig, then it specifies a downstream
    node's activation, which will be used to compute a gradient. If it is None, it is assumed that the output logits (in
    their entirety) are being reconstructed instead, and a loss function computed, to compute the gradient.
    """
    assert trace_config.attention_trace_type == AttentionTraceType.V
    if trace_config.downstream_trace_config is None:
        assert loss_fn_for_backward_pass is not None
        assert isinstance(model_context, StandardModelContext)  # for typechecking
        logit_reconstituter = LogitReconstituter(
            model_context=model_context,
            detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE,
        )
        downstream_activation_index_for_reconstituter = (
            logit_reconstituter.get_residual_activation_index()
        )

        def reconstitute_gradient_fn(downstream_resid: torch.Tensor) -> torch.Tensor:
            reconstitute_gradient = logit_reconstituter.make_reconstitute_gradient_of_loss_fn(
                loss_fn=loss_fn_for_backward_pass
            )
            return reconstitute_gradient(downstream_resid)

    else:
        if trace_config.node_type.is_autoencoder_latent:
            assert multi_autoencoder_context is not None
            autoencoder_context = multi_autoencoder_context.get_autoencoder_context(
                trace_config.node_type
            )
            assert autoencoder_context is not None
        else:
            autoencoder_context = None
        act_reconstituter = ActivationReconstituter.from_trace_config(
            transformer=model_context.get_or_create_model(),
            autoencoder_context=autoencoder_context,
            trace_config=trace_config.downstream_trace_config,
        )
        downstream_trace_config = trace_config.downstream_trace_config
        downstream_activation_index_for_reconstituter = act_reconstituter.get_residual_activation_index_for_node_index(
            # convert trace_config to node index
            downstream_trace_config.node_index
        )

        def reconstitute_gradient_fn(
            downstream_resid: torch.Tensor,
        ) -> torch.Tensor:
            reconstitute_gradient_with_args = (
                act_reconstituter.make_reconstitute_gradient_fn_for_trace_config(
                    trace_config=downstream_trace_config
                )
            )
            return reconstitute_gradient_with_args(
                downstream_resid,
                downstream_trace_config.layer_index,
                downstream_trace_config.pass_type,
            )

    assert trace_config.layer_index is not None
    direct_effect_reconstituter = AttentionDirectEffectReconstituter(
        model_context=model_context,
        layer_indices=[trace_config.layer_index],
        detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE,
    )
    upstream_activation_index_for_reconstituter = direct_effect_reconstituter.get_residual_activation_index_for_node_index(
        # convert trace_config to node index
        trace_config.node_index
    )

    upstream_scalar_hook = direct_effect_reconstituter.make_scalar_hook_for_node_index(
        trace_config.node_index
    )

    def reconstitute_direct_effect_fn(
        upstream_resid: torch.Tensor,
        downstream_resid: torch.Tensor | None,
    ) -> torch.Tensor:
        assert downstream_resid is not None
        gradient = reconstitute_gradient_fn(downstream_resid).detach()
        activations = direct_effect_reconstituter.reconstitute_activations(
            resid=upstream_resid,
            grad=gradient,
            layer_index=trace_config.layer_index,
            pass_type=trace_config.pass_type,
        )
        return upstream_scalar_hook(activations)

    return (
        upstream_activation_index_for_reconstituter,
        downstream_activation_index_for_reconstituter,
        reconstitute_direct_effect_fn,
    )


def replace_activation_index_using_reconstituter(
    model_context: ModelContext,
    multi_autoencoder_context: MultiAutoencoderContext | None,
    batched_ds_computation_params: list[DerivedScalarComputationParams],
) -> tuple[
    list[ActivationIndex | None],
    list[ActivationIndex | None],
    list[Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]],
]:
    """
    Where trace_config occurs in batched_ds_computation_params, convert it to the
    upstream_activation_index (and optionally also downstream_activation_index)
    corresponding to the preceding residual stream required by a Reconstituter. Also
    return a function, generated from the Reconstituter, to obtain the activation
    corresponding to the original trace_config from the residual stream.
    """
    batched_reconstitute_activation_fn: list[
        Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
    ] = []
    batched_upstream_activation_index_to_grab: list[ActivationIndex | None] = []
    batched_downstream_activation_index_to_grab: list[ActivationIndex | None] = []
    for ds_computation_params_index in range(len(batched_ds_computation_params)):
        ds_computation_params = batched_ds_computation_params[ds_computation_params_index]
        trace_config = ds_computation_params.trace_config
        if trace_config is not None:
            if trace_config.attention_trace_type == AttentionTraceType.V:
                (
                    upstream_activation_index_for_reconstituter,
                    downstream_activation_index_for_reconstituter,
                    reconstitute_activation_fn,
                ) = get_activation_indices_and_reconstitute_direct_effect_fn(
                    model_context=model_context,
                    multi_autoencoder_context=multi_autoencoder_context,
                    trace_config=trace_config,
                    loss_fn_for_backward_pass=ds_computation_params.loss_fn_for_backward_pass,
                )
            else:
                (
                    upstream_activation_index_for_reconstituter,
                    reconstitute_activation_fn,
                ) = get_activation_index_and_reconstitute_activation_fn(
                    transformer=model_context.get_or_create_model(),
                    multi_autoencoder_context=multi_autoencoder_context,
                    trace_config=trace_config,
                )
                downstream_activation_index_for_reconstituter = None
        else:
            upstream_activation_index_for_reconstituter = None
            downstream_activation_index_for_reconstituter = None

            def dummy_reconstitute_activation_fn(
                resid: torch.Tensor, grad: torch.Tensor | None
            ) -> torch.Tensor:
                raise NotImplementedError("This function should not be called")

            reconstitute_activation_fn = dummy_reconstitute_activation_fn
        if upstream_activation_index_for_reconstituter is not None:
            assert (
                upstream_activation_index_for_reconstituter.activation_location_type.node_type
                == NodeType.RESIDUAL_STREAM_CHANNEL
            )
        if downstream_activation_index_for_reconstituter is not None:
            assert (
                downstream_activation_index_for_reconstituter.activation_location_type.node_type
                == NodeType.RESIDUAL_STREAM_CHANNEL
            )
        batched_upstream_activation_index_to_grab.append(
            upstream_activation_index_for_reconstituter
        )
        batched_downstream_activation_index_to_grab.append(
            downstream_activation_index_for_reconstituter
        )
        batched_reconstitute_activation_fn.append(reconstitute_activation_fn)
    return (
        batched_upstream_activation_index_to_grab,
        batched_downstream_activation_index_to_grab,
        batched_reconstitute_activation_fn,
    )


def run_inference_and_populate_raw_store(
    model_context: StandardModelContext,
    multi_autoencoder_context: MultiAutoencoderContext | None,
    batched_ds_computation_params: list[DerivedScalarComputationParams],
) -> tuple[
    list[RawActivationStore],
    list[float | None],
    list[float | None],
    list[float | None],
    float,
]:
    """
    This populates a dict of ActivationsAndMetadata objects for each batch element, and returns
    inference-related stats.

     - batched_requested_activations_by_location_type_and_pass_type: stored activations
     - batched_loss_floats: loss values for each batch element, if a loss function was specified
     - batched_activation_value_for_backward_pass_floats: value of activation for which a backward pass was
    computed for each batch element, if an activation index for backward pass was specified
     - batched_memory_used_before: amount of memory allocated to the GPU before the forward pass for each batch element
     - inference_time: time used for the forward pass, in seconds
    """
    for params in batched_ds_computation_params:
        trace_config = params.trace_config
        if trace_config is not None:
            if trace_config.node_type.is_autoencoder_latent:
                assert multi_autoencoder_context is not None
                assert (
                    multi_autoencoder_context.get_autoencoder_context(trace_config.node_type)
                    is not None
                ), f"Autoencoder context not found for {trace_config.node_type}"

    transformer = model_context.get_or_create_model()
    batched_input_token_ints = [params.input_token_ints for params in batched_ds_computation_params]
    tokens_tensor, pad_tensor = prep_input_and_right_pad_for_forward_pass(
        batched_input_token_ints, transformer.device
    )

    (
        batched_upstream_activation_index_for_backward_pass,
        batched_downstream_activation_index_for_backward_pass,
        batched_reconstitute_activation_fn,
    ) = replace_activation_index_using_reconstituter(
        model_context=model_context,
        multi_autoencoder_context=multi_autoencoder_context,
        batched_ds_computation_params=batched_ds_computation_params,
    )

    batched_activation_index_for_backward_pass_by_name = [
        {
            "upstream": upstream_activation_index_for_backward_pass,
            "downstream": downstream_activation_index_for_backward_pass,
        }
        for (
            upstream_activation_index_for_backward_pass,
            downstream_activation_index_for_backward_pass,
        ) in zip(
            batched_upstream_activation_index_for_backward_pass,
            batched_downstream_activation_index_for_backward_pass,
        )
    ]
    (
        transformer_graph,  # Stores the hooks
        batched_requested_activations_by_location_type_and_pass_type,  # Stores the activations from the hooks after the forward pass
        batched_requested_attached_activations_for_backward_pass_by_name,
    ) = get_transformer_graph_hooks_and_activation_caches(
        multi_autoencoder_context=multi_autoencoder_context,
        batched_ds_computation_params=batched_ds_computation_params,
        batched_activation_index_for_backward_pass_by_name=batched_activation_index_for_backward_pass_by_name,
    )

    assert len(batched_requested_activations_by_location_type_and_pass_type) == len(
        batched_ds_computation_params
    )
    for (
        requested_activations_by_location_type_and_pass_type,
        ds_computation_params,
    ) in zip(
        batched_requested_activations_by_location_type_and_pass_type,
        batched_ds_computation_params,
    ):
        pass_types = [
            activation_location_type_and_pass_type.pass_type
            for activation_location_type_and_pass_type in requested_activations_by_location_type_and_pass_type.keys()
        ]

        if any(pass_type == PassType.BACKWARD for pass_type in pass_types):
            assert (
                ds_computation_params.loss_fn_for_backward_pass is not None
                or ds_computation_params.trace_config is not None
            ), "loss_fn_for_backward_pass or trace_config must be defined if gradients are required"

    batched_device_for_raw_activations = [
        params.device_for_raw_activations for params in batched_ds_computation_params
    ]

    t0 = time.time()
    cuda_available = torch.cuda.is_available()
    if cuda_available and any(
        device.type == "cuda" for device in batched_device_for_raw_activations
    ):
        torch.cuda.empty_cache()
    batched_memory_used_before: list[float | None] = [
        torch.cuda.memory_allocated(device) if device.type == "cuda" and cuda_available else None
        for device in batched_device_for_raw_activations
    ]

    logits, _ = transformer.forward(
        tokens_tensor, pad=pad_tensor, hooks=transformer_graph.as_transformer_hooks()
    )
    batched_loss: list[torch.Tensor | None] = []
    batched_activation_value_for_backward_pass: list[torch.Tensor | None] = []
    for batch_index, (
        ds_computation_params,
        requested_attached_activation_for_backward_pass_by_name,
        reconstitute_activation_fn,
        activation_index_for_backward_pass_by_name,
    ) in enumerate(
        zip(
            batched_ds_computation_params,
            batched_requested_attached_activations_for_backward_pass_by_name,
            batched_reconstitute_activation_fn,
            batched_activation_index_for_backward_pass_by_name,
        )
    ):
        loss_fn_for_backward_pass = ds_computation_params.loss_fn_for_backward_pass
        if loss_fn_for_backward_pass is not None:
            loss = loss_fn_for_backward_pass(logits[batch_index].unsqueeze(0))
        else:
            loss = None

        if activation_index_for_backward_pass_by_name["upstream"] is not None:
            assert requested_attached_activation_for_backward_pass_by_name["upstream"] is not None
            activation_value_for_backward_pass = reconstitute_activation_fn(
                requested_attached_activation_for_backward_pass_by_name["upstream"],
                requested_attached_activation_for_backward_pass_by_name["downstream"],
            )
        else:
            activation_value_for_backward_pass = None

        batched_loss.append(loss)
        batched_activation_value_for_backward_pass.append(activation_value_for_backward_pass)

    populated_losses: list[torch.Tensor] = []
    for loss, value in zip(batched_loss, batched_activation_value_for_backward_pass):
        # backward pass is computed from value if it is not None, otherwise from loss
        if value is not None:
            populated_losses.append(value)
        elif loss is not None:
            populated_losses.append(loss)

    if len(populated_losses):
        assert all(isinstance(loss, torch.Tensor) for loss in populated_losses)
        loss_sum = sum(populated_losses)
        assert isinstance(loss_sum, torch.Tensor)
        loss_sum.backward()

    inference_time = time.time() - t0

    batched_loss_floats: list[float | None] = [
        loss.item() if loss is not None and not torch.isnan(loss) else None for loss in batched_loss
    ]

    batched_activation_value_for_backward_pass_floats: list[float | None] = [
        activation.item() if activation is not None else None
        for activation in batched_activation_value_for_backward_pass
    ]

    batched_raw_activation_store: list[RawActivationStore] = []
    for (requested_activations_by_location_type_and_pass_type,) in zip(
        batched_requested_activations_by_location_type_and_pass_type,
    ):
        raw_activation_store = RawActivationStore.from_nested_dict_of_activations(
            requested_activations_by_location_type_and_pass_type
        )
        batched_raw_activation_store.append(raw_activation_store)

    assert (
        len(batched_raw_activation_store)
        == len(batched_loss_floats)
        == len(batched_activation_value_for_backward_pass_floats)
        == len(batched_memory_used_before)
        == len(batched_ds_computation_params)
    )  # returns one batch element per input param setting

    return (
        batched_raw_activation_store,
        batched_loss_floats,
        batched_activation_value_for_backward_pass_floats,
        batched_memory_used_before,
        inference_time,
    )


def construct_ds_stores_from_raw(
    batched_raw_activation_store: list[RawActivationStore],
    batched_ds_computation_params: list[DerivedScalarComputationParams],
) -> tuple[list[dict[str, MultiGroupDerivedScalarStore]], list[float | None]]:
    batched_multi_group_ds_store_by_processing_step: list[
        dict[str, MultiGroupDerivedScalarStore]
    ] = []
    batched_memory_used_after: list[float | None] = []
    for (
        raw_activation_store,
        ds_computation_params,
    ) in zip(
        batched_raw_activation_store,
        batched_ds_computation_params,
    ):
        multi_group_scalar_derivers_by_processing_step = (
            ds_computation_params.multi_group_scalar_derivers_by_processing_step
        )
        multi_group_ds_store_by_processing_step: dict[str, MultiGroupDerivedScalarStore] = {}
        for (
            spec_name,
            multi_group_scalar_derivers,
        ) in multi_group_scalar_derivers_by_processing_step.items():
            multi_group_ds_store_by_processing_step[
                spec_name
            ] = MultiGroupDerivedScalarStore.derive_from_raw(
                raw_activation_store, multi_group_scalar_derivers
            )
        batched_multi_group_ds_store_by_processing_step.append(
            multi_group_ds_store_by_processing_step
        )

        device_for_raw_activations = ds_computation_params.device_for_raw_activations
        memory_used_after = None
        if torch.cuda.is_available() and device_for_raw_activations.type == "cuda":
            gc.collect()
            memory_used_after = torch.cuda.memory_allocated(device_for_raw_activations)
        batched_memory_used_after.append(memory_used_after)

    return (
        batched_multi_group_ds_store_by_processing_step,
        batched_memory_used_after,
    )


def get_transformer_graph_hooks_and_activation_caches(
    multi_autoencoder_context: MultiAutoencoderContext | None,
    batched_ds_computation_params: list[DerivedScalarComputationParams],
    batched_activation_index_for_backward_pass_by_name: list[dict[str, ActivationIndex | None]],
) -> tuple[
    TransformerHookGraph,
    list[dict[ActivationLocationTypeAndPassType, dict[LayerIndex, torch.Tensor]]],
    list[dict[str, torch.Tensor | None]],
]:
    """This is a helper function that returns:
    1. a TransformerHookGraph object containing hooks for the given
    scalar derivers (this can be passed to Transformer using the as_transformer_hooks method to add
    hooks to the transformer forward and backward pass)
    2. dictionaries mapping each activation location type to the activations requested
    (before it is filled with activations during forward passes), one dictionary per batch element
    3. dictionaries each containing just one value, the scalar tensor on which the backward pass can be run (
    unlike the activations in 2, this tensor is still attached to the pytorch model), one dictionary per batch element
    """

    batched_activation_location_type_and_pass_types = [
        params.activation_location_type_and_pass_types for params in batched_ds_computation_params
    ]
    batched_device = [params.device_for_raw_activations for params in batched_ds_computation_params]
    batched_ablation_specs = [params.ablation_specs for params in batched_ds_computation_params]
    batched_prompt_lengths = [params.prompt_length for params in batched_ds_computation_params]

    # This is a callable that can be used similarly to a Hooks object.
    transformer_graph = TransformerHookGraph()

    """This step constructs the activation location types needed (for the case where they don't already exist). Any hooks to be added
    to that location type can then be appended to transformer_graph in the normal way.
    """

    # steps:
    # 1. add autoencoder graph to transformer_graph (injected autoencoders specified in multi_autoencoder_context are
    #    hooked in a second set of forward hooks, called after ablating and saving hooks, and before activation grabbing hooks)
    #
    # for each batched element:
    #     2. add ablating hooks
    #     3. add activation grabbing hooks (storing without detaching, for backward pass). These come last, after ablating and saving hooks.
    #     4. add saving hooks (storing with detaching)
    #
    # Because the autoencoder is hooked in a second set of hooks, followed by the grabbing hooks, they are always called last:
    # - fwd hooks: ablate activations, save activations, ...
    # - bwd hooks: ablate gradients, save gradients, ...
    # - fwd2 hooks: autoencoder (convert to latent, ablate latents, grab latents, save latents, convert back to activations), grab activations, ...

    # add autoencoder graph
    if multi_autoencoder_context is not None:
        has_multiple_autoencoders = (
            len(multi_autoencoder_context.autoencoder_context_by_node_type) > 1
        )
        for (
            node_type,
            autoencoder_context,
        ) in multi_autoencoder_context.autoencoder_context_by_node_type.items():
            subgraph = AutoencoderHookGraph(
                autoencoder_context, is_one_of_multiple_autoencoders=has_multiple_autoencoders
            )
            subgraph_name = f"{node_type.value}"
            transformer_graph.inject_subgraph(subgraph, subgraph_name)

    (
        batched_requested_activations_by_location_type_and_pass_type,
        batched_requested_attached_activation_for_backward_pass_by_name,
    ) = ([], [])
    assert (
        len(batched_activation_location_type_and_pass_types)
        == len(batched_ablation_specs)
        == len(batched_activation_index_for_backward_pass_by_name)
        == len(batched_prompt_lengths)
    )
    for i in range(len(batched_activation_location_type_and_pass_types)):
        # add ablating hooks
        ablation_spec = batched_ablation_specs[i]
        if ablation_spec is not None:
            add_ablating_hooks(transformer_graph, ablation_spec, batch_index=i)

        # add activation grabbing hooks; the grabbed activations are stored in dicts, keyed by the
        # string name assigned to them
        requested_attached_activation_for_backward_pass_by_name: dict[str, torch.Tensor | None] = {}
        for (
            name,
            activation_index_for_backward_pass,
        ) in batched_activation_index_for_backward_pass_by_name[i].items():
            if activation_index_for_backward_pass is not None:
                requested_attached_activation_for_backward_pass_by_name = add_grabbing_hook_for_backward_pass(
                    requested_attached_activation_for_backward_pass_by_name,
                    name,
                    transformer_graph=transformer_graph,
                    activation_index_for_backward_pass=activation_index_for_backward_pass,
                    batch_index=i,
                    append_to_fwd2=True,  # append to fwd2 when using Reconstituter to obtain gradients, so that
                    # the preceding residual stream backward pass hooks can be called after running .backward()
                    # from the grabbed activation
                )
            else:
                requested_attached_activation_for_backward_pass_by_name[name] = None

        # add saving hooks
        requested_activations_by_location_type_and_pass_type = add_saving_hooks(
            transformer_graph=transformer_graph,
            activation_location_type_and_pass_types=batched_activation_location_type_and_pass_types[
                i
            ],
            device=batched_device[i],
            unpadded_prompt_length=batched_prompt_lengths[i],
            batch_index=i,
        )

        batched_requested_activations_by_location_type_and_pass_type.append(
            requested_activations_by_location_type_and_pass_type
        )
        batched_requested_attached_activation_for_backward_pass_by_name.append(
            requested_attached_activation_for_backward_pass_by_name
        )

    return (
        transformer_graph,
        batched_requested_activations_by_location_type_and_pass_type,
        batched_requested_attached_activation_for_backward_pass_by_name,
    )


def create_activation_grabbing_hook_fn(
    attached_activation_dict: dict[str, torch.Tensor | None],
    name: str,
    activation_index: ActivationIndex,
    batch_index: int,
) -> tuple[Callable, ActivationLocationTypeAndPassType, LayerIndex, dict[str, torch.Tensor | None]]:
    assert (
        name not in attached_activation_dict
    ), f"Name {name} already exists in attached_activation_dict"

    def activation_grabbing_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        indices: tuple[slice | int, ...] = (
            batch_index,
        ) + make_python_slice_from_all_or_one_indices(
            activation_index.tensor_indices
        )  # use the batch_index to index into the batch dimension
        attached_activation_dict[name] = act[indices]  # .clone()
        return act

    return (
        activation_grabbing_hook_fn,
        ActivationLocationTypeAndPassType(
            activation_location_type=activation_index.activation_location_type,
            pass_type=activation_index.pass_type,
        ),
        activation_index.layer_index,
        attached_activation_dict,
    )


def add_grabbing_hook_for_backward_pass(
    attached_activation_dict: dict[str, torch.Tensor | None],
    name: str,
    transformer_graph: TransformerHookGraph,
    activation_index_for_backward_pass: ActivationIndex,
    batch_index: int,
    append_to_fwd2: bool = False,
) -> dict[str, torch.Tensor | None]:
    """This is a helper function that returns a TransformerHooks object containing hooks at naturally existing hook locations
    (e.g. MLP post-activations, rather than autoencoder latent activations) for the given
    scalar derivers, and a dictionary mapping each activation location type with a naturally existing hook to the activations requested
    (before it is filled with activations during forward passes)."""

    (
        hook_fn,
        activation_location_type_and_pass_type,
        layer_index,
        attached_activation_dict,
    ) = create_activation_grabbing_hook_fn(
        attached_activation_dict, name, activation_index_for_backward_pass, batch_index=batch_index
    )

    transformer_graph.append(
        activation_location_type_and_pass_type,
        hook_fn,
        layer_indices=layer_index,
        append_to_fwd2=append_to_fwd2,
    )

    return attached_activation_dict


def create_ablating_hook_fn(
    ablation_spec: AblationSpec,
    batch_index: int,
) -> tuple[Callable, ActivationLocationTypeAndPassType, LayerIndex]:
    activation_location_type_and_pass_type = ActivationLocationTypeAndPassType(
        activation_location_type=ablation_spec.index.activation_location_type,
        pass_type=ablation_spec.index.pass_type,
    )

    layer_index = ablation_spec.index.layer_index

    def ablating_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Initial dimension is batch
        act = act.clone()
        python_slice_operator: tuple[slice | int, ...] = (
            slice(batch_index, batch_index + 1),
        ) + make_python_slice_from_all_or_one_indices(ablation_spec.index.tensor_indices)
        assert len(python_slice_operator) == len(act.shape), (
            len(python_slice_operator),
            python_slice_operator,
            len(act.shape),
            act.shape,
        )
        act[python_slice_operator] = ablation_spec.value
        return act

    return ablating_hook_fn, activation_location_type_and_pass_type, layer_index


def add_ablating_hooks(
    transformer_graph: TransformerHookGraph,
    ablation_specs: list[AblationSpec],
    batch_index: int,
) -> None:
    """This is a helper function that returns a TransformerHooks object containing hooks at naturally existing hook locations
    (e.g. MLP post-activations, rather than autoencoder latent activations) for the given
    scalar derivers, and a dictionary mapping each activation location type with a naturally existing hook to the activations requested
    (before it is filled with activations during forward passes)."""

    for ablation_spec in ablation_specs:
        (
            hook_fn,
            activation_location_type_and_pass_type,
            layer_index,
        ) = create_ablating_hook_fn(ablation_spec, batch_index=batch_index)

        transformer_graph.append(
            activation_location_type_and_pass_type,
            hook_fn,
            layer_indices=layer_index,
        )


def create_saving_hook_fn(
    device: torch.device,
    activation_location_type_and_pass_type: ActivationLocationTypeAndPassType,
    unpadded_prompt_length: int,
    batch_index: int,
) -> tuple[Callable, dict[LayerIndex, torch.Tensor]]:
    requested_activations_by_layer_index = {}

    def saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        layer_index = kwargs.get("layer", None)
        # First dimension is batch, second dimension is sequence length. We truncate the sequence
        # length to the unpadded prompt length. If the third dimension is also sequence length, we
        # truncate that too.
        shape_spec = (
            activation_location_type_and_pass_type.activation_location_type.shape_spec_per_token_sequence
        )

        def get_slice_for_dim(dim: Dimension) -> slice:
            if dim.is_sequence_token_dimension:
                return slice(None, unpadded_prompt_length)
            else:
                return slice(None)

        truncated_act = act[(batch_index,) + tuple([get_slice_for_dim(dim) for dim in shape_spec])]
        requested_activations_by_layer_index[layer_index] = truncated_act.detach().to(device)
        return act

    return saving_hook_fn, requested_activations_by_layer_index


def add_saving_hooks(
    transformer_graph: TransformerHookGraph,
    activation_location_type_and_pass_types: list[ActivationLocationTypeAndPassType],
    device: torch.device,
    unpadded_prompt_length: int,
    batch_index: int,
) -> dict[ActivationLocationTypeAndPassType, dict[LayerIndex, torch.Tensor]]:
    """This is a helper function that returns a TransformerHooks object containing hooks at naturally existing hook locations
    (e.g. MLP post-activations, rather than autoencoder latent activations) for the given
    scalar derivers, and a dictionary mapping each activation location type with a naturally existing hook to the activations requested
    (before it is filled with activations during forward passes)."""

    requested_activations_by_location_type_and_pass_type = {}

    for activation_location_type_and_pass_type in activation_location_type_and_pass_types:
        (
            hook_fn,
            requested_activations_by_location_type_and_pass_type[
                activation_location_type_and_pass_type
            ],
        ) = create_saving_hook_fn(
            device,
            activation_location_type_and_pass_type=activation_location_type_and_pass_type,
            unpadded_prompt_length=unpadded_prompt_length,
            batch_index=batch_index,
        )

        transformer_graph.append(
            activation_location_type_and_pass_type,
            hook_fn,
        )

    return requested_activations_by_location_type_and_pass_type


def apply_default_dst_configs_to_dst_and_config_list(
    model_context: StandardModelContext,
    multi_autoencoder_context: MultiAutoencoderContext | None,
    dst_and_config_list: list[tuple[DerivedScalarType, DstConfig | None]],
) -> list[tuple[DerivedScalarType, DstConfig]]:
    def get_default_dst_config(
        dst: DerivedScalarType,
    ) -> DstConfig:
        return DstConfig(
            model_context=model_context,
            multi_autoencoder_context=multi_autoencoder_context,
            derive_gradients=not dst.requires_grad_for_forward_pass,
        )

    return [
        (dst, config if config is not None else get_default_dst_config(dst))
        for dst, config in dst_and_config_list
    ]


def get_ds_computation_params_for_prompt(
    model_context: StandardModelContext,
    autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None,
    dst_and_config_list: list[
        tuple[DerivedScalarType, DstConfig | None]
    ],  # None -> default config for dst
    prompt: str,
    loss_fn_for_backward_pass: Callable[[torch.Tensor], torch.Tensor] | None,
    trace_config: TraceConfig | None,
    ablation_specs: list[AblationSpec],
) -> DerivedScalarComputationParams:
    assert (loss_fn_for_backward_pass is None) or (trace_config is None)
    multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
        autoencoder_context
    )

    dst_and_config_list_with_default_config = apply_default_dst_configs_to_dst_and_config_list(
        model_context, multi_autoencoder_context, dst_and_config_list
    )

    multi_group_scalar_derivers_by_processing_step = {
        "dummy": MultiGroupScalarDerivers.from_dst_and_config_list(
            dst_and_config_list_with_default_config
        )  # "dummy" is a placeholder processing step name
    }

    input_token_ints = model_context.encode(prompt)

    return DerivedScalarComputationParams(
        input_token_ints=input_token_ints,
        multi_group_scalar_derivers_by_processing_step=multi_group_scalar_derivers_by_processing_step,
        loss_fn_for_backward_pass=loss_fn_for_backward_pass,
        device_for_raw_activations=model_context.device,
        ablation_specs=ablation_specs,
        trace_config=trace_config,
    )


def get_derived_scalars_for_prompt(
    model_context: StandardModelContext,
    dst_and_config_list: list[
        tuple[DerivedScalarType, DstConfig | None]
    ],  # None -> default config for dst
    prompt: str,
    loss_fn_for_backward_pass: Callable[[torch.Tensor], torch.Tensor] | None = None,
    trace_config: TraceConfig | None = None,
    autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None = None,
    ablation_specs: list[AblationSpec] = [],
) -> tuple[DerivedScalarStore, InferenceAndTokenData, RawActivationStore]:
    """
    Lightweight function to populate a DerivedScalarStore given information specifying the prompt, loss function, and derived scalars to compute.
    """
    multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
        autoencoder_context
    )

    input_token_ints = model_context.encode(prompt)
    input_token_strings = [model_context.decode_token(token_int) for token_int in input_token_ints]

    dst_and_config_list_with_default_config = apply_default_dst_configs_to_dst_and_config_list(
        model_context, multi_autoencoder_context, dst_and_config_list
    )

    multi_group_scalar_derivers_by_processing_step = {
        "dummy": MultiGroupScalarDerivers.from_dst_and_config_list(
            dst_and_config_list_with_default_config
        )
    }  # "dummy" is a placeholder processing step name

    ds_computation_params = DerivedScalarComputationParams(
        input_token_ints=input_token_ints,
        multi_group_scalar_derivers_by_processing_step=multi_group_scalar_derivers_by_processing_step,
        loss_fn_for_backward_pass=loss_fn_for_backward_pass,
        device_for_raw_activations=model_context.device,
        trace_config=trace_config,
        ablation_specs=ablation_specs,
    )
    batched_multi_group_ds_store_by_processing_step: list[dict[str, MultiGroupDerivedScalarStore]]
    (
        batched_multi_group_ds_store_by_processing_step,
        batched_inference_data,
        batched_raw_activation_store,
    ) = compute_derived_scalar_groups_for_input_token_ints(
        model_context=model_context,
        multi_autoencoder_context=multi_autoencoder_context,
        batched_ds_computation_params=[ds_computation_params],
    )
    ds_store = batched_multi_group_ds_store_by_processing_step[0]["dummy"].to_single_ds_store()
    inference_data = batched_inference_data[0]
    inference_and_token_data = InferenceAndTokenData(
        **inference_data.dict(),
        tokens_as_ints=input_token_ints,
        tokens_as_strings=input_token_strings,
    )
    raw_activation_store = batched_raw_activation_store[0]

    return ds_store, inference_and_token_data, raw_activation_store


def get_batched_derived_scalars_for_prompt(
    model_context: StandardModelContext,
    batched_dst_and_config_list: list[
        list[tuple[DerivedScalarType, DstConfig | None]]
    ],  # None -> default config for dst
    batched_prompt: list[str],
    loss_fn_for_backward_pass: Callable[[torch.Tensor], torch.Tensor] | None = None,
    trace_config: TraceConfig | None = None,
    autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None = None,
    ablation_specs: list[AblationSpec] = [],
) -> tuple[list[DerivedScalarStore], list[InferenceAndTokenData], list[RawActivationStore]]:
    """
    Lightweight function to populate a DerivedScalarStore given information specifying the prompt, loss function, and derived scalars to compute.
    """
    multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
        autoencoder_context
    )

    assert len(batched_dst_and_config_list) == len(batched_prompt)

    batched_ds_computation_params = [
        get_ds_computation_params_for_prompt(
            model_context=model_context,
            autoencoder_context=autoencoder_context,
            dst_and_config_list=dst_and_config_list,
            prompt=prompt,
            loss_fn_for_backward_pass=loss_fn_for_backward_pass,
            trace_config=trace_config,
            ablation_specs=ablation_specs,
        )
        for prompt, dst_and_config_list in zip(batched_prompt, batched_dst_and_config_list)
    ]

    batched_multi_group_ds_store_by_processing_step: list[dict[str, MultiGroupDerivedScalarStore]]
    (
        batched_multi_group_ds_store_by_processing_step,
        batched_inference_data,
        batched_raw_activation_store,
    ) = compute_derived_scalar_groups_for_input_token_ints(
        model_context=model_context,
        multi_autoencoder_context=multi_autoencoder_context,
        batched_ds_computation_params=batched_ds_computation_params,
    )
    batched_ds_store = [
        multi_group_ds_store["dummy"].to_single_ds_store()
        for multi_group_ds_store in batched_multi_group_ds_store_by_processing_step
    ]
    batched_inference_and_token_data = []
    for ds_computation_params, inference_data in zip(
        batched_ds_computation_params, batched_inference_data
    ):
        input_token_ints = ds_computation_params.input_token_ints
        input_token_strings = [
            model_context.decode_token(token_int) for token_int in input_token_ints
        ]
        batched_inference_and_token_data.append(
            InferenceAndTokenData(
                **inference_data.dict(),
                tokens_as_ints=input_token_ints,
                tokens_as_strings=input_token_strings,
            )
        )

    return batched_ds_store, batched_inference_and_token_data, batched_raw_activation_store
