import math
from typing import Any, TypeVar

import numpy as np
import pydantic
import torch

from neuron_explainer.activation_server.derived_scalar_computation import (
    LossFnConfig,
    LossFnName,
    get_derived_scalars_for_prompt,
    maybe_construct_loss_fn_for_backward_pass,
)
from neuron_explainer.activation_server.interactive_model import (
    AblationSpec,
    BatchedRequest,
    InferenceRequestSpec,
    InferenceSubRequest,
    InteractiveModel,
    MultipleTopKDerivedScalarsRequest,
    MultipleTopKDerivedScalarsRequestSpec,
    MultipleTopKDerivedScalarsResponseData,
    Tensor0D,
    Tensor1D,
    Tensor2D,
)
from neuron_explainer.activation_server.requests_and_responses import (
    BatchedResponse,
    BatchedTdbRequest,
    ComponentTypeForAttention,
    ComponentTypeForMlp,
    DerivedScalarsRequestSpec,
    GroupId,
    ProcessingRequestSpec,
    ScoredTokensRequestSpec,
    TdbRequestSpec,
    TokenScoringType,
)
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    ActivationIndex,
    AttentionTraceType,
    DerivedScalarIndex,
    MirroredActivationIndex,
    MirroredNodeIndex,
    MirroredTraceConfig,
    NodeToTrace,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.postprocessing import TokenWriteConverter
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.hooks import AtLayers, TransformerHooks
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    NodeType,
    PassType,
)
from neuron_explainer.models.model_context import StandardModelContext
from neuron_explainer.tests.conftest import AUTOENCODER_TEST_DST
from neuron_explainer.tests.test_all_dsts import DETACH_LAYER_NORM_SCALE_FOR_TEST

dst_by_group_id_and_component_type: dict[GroupId, dict[str, DerivedScalarType]] = {
    GroupId.WRITE_NORM: {
        "mlp": DerivedScalarType.MLP_WRITE_NORM,
        "autoencoder": DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM,
        "unflattened_attn": DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM,
        "per_token_attn": DerivedScalarType.ATTN_WRITE_NORM_PER_SEQUENCE_TOKEN,
    },
    GroupId.ACT_TIMES_GRAD: {
        "mlp": DerivedScalarType.MLP_ACT_TIMES_GRAD,
        "autoencoder": DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD,
        "unflattened_attn": DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD,
        "per_token_attn": DerivedScalarType.ATTN_ACT_TIMES_GRAD_PER_SEQUENCE_TOKEN,
    },
    GroupId.DIRECT_WRITE_TO_GRAD: {
        "mlp": DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD,
        "autoencoder": DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
        "unflattened_attn": DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
        "per_token_attn": DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD_PER_SEQUENCE_TOKEN,
    },
    GroupId.ACTIVATION: {
        "mlp": DerivedScalarType.MLP_POST_ACT,
        "autoencoder": DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
        "unflattened_attn": DerivedScalarType.ATTN_QK_PROBS,
    },
}


def make_dst_list_by_group_id(
    group_ids: list[GroupId],
    component_types: list[str],
) -> dict[GroupId, list[DerivedScalarType]]:
    dst_list_by_group_id: dict[GroupId, list[DerivedScalarType]] = {}
    for group_id in group_ids:
        assert group_id in dst_by_group_id_and_component_type
        dst_list_by_group_id[group_id] = []
        for component_type in component_types:
            assert component_type in dst_by_group_id_and_component_type[group_id]
            dst = dst_by_group_id_and_component_type[group_id][component_type]
            dst_list_by_group_id[group_id].append(dst)
    return dst_list_by_group_id


def assert_acts_within_epsilon(
    acts1: list[float], acts2: list[float], epsilon: float = 1e-3
) -> None:
    assert len(acts1) == len(acts2), "Activations lists are of different lengths"
    for a, b in zip(acts1, acts2):
        assert abs(a - b) < epsilon, f"Pair of activations differ by more than {epsilon}"


def assert_common_acts_within_epsilon(
    acts1: list[float],
    acts2: list[float],
    # indices1 must be a subset of indices2
    indices1: list[MirroredNodeIndex],
    indices2: list[MirroredNodeIndex],
    # some benign numeric inconsistencies appear to cause diffs just a bit over 1e-3
    epsilon: float = 2e-3,
) -> None:
    assert set(indices1).issubset(set(indices2)), f"{indices1=} is not a subset of {indices2=}"
    for i, node_index in enumerate(indices1):
        j = indices2.index(node_index)
        assert (
            abs(acts1[i] - acts2[j]) < epsilon
        ), f"Pair of activations differ by more than {epsilon}"


async def test_forward_and_backward_pass_request(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context,
        standard_autoencoder_context,
    )

    loss_fn_config = LossFnConfig(
        name=LossFnName.LOGIT_DIFF,
        target_tokens=["!"],
        distractor_tokens=["."],
    )

    # test per-sequence-token attention DSTs in multi-request context
    multi_top_k_request = MultipleTopKDerivedScalarsRequest(
        inference_request_spec=InferenceRequestSpec(
            prompt="Hello world",
            loss_fn_config=loss_fn_config,
        ),
        multiple_top_k_derived_scalars_request_spec=MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id=make_dst_list_by_group_id(
                group_ids=[
                    GroupId.WRITE_NORM,
                    GroupId.ACT_TIMES_GRAD,
                    GroupId.DIRECT_WRITE_TO_GRAD,
                ],
                component_types=["mlp", "per_token_attn"],
            ),
            # dsts for each group ID are assumed to have defined node_type,
            # all node_types assumed to be distinct within a group_id, and all group_ids to
            # contain the same set of node_types.
            token_index=None,
            top_and_bottom_k=10,
            pass_type=PassType.FORWARD,
        ),
    )

    # test layer requests with backward pass from activation
    single_request = MultipleTopKDerivedScalarsRequest(
        inference_request_spec=InferenceRequestSpec(
            prompt="Hello world",
            loss_fn_config=None,
            trace_config=MirroredTraceConfig.from_trace_config(
                TraceConfig.from_activation_index(
                    ActivationIndex(
                        activation_location_type=ActivationLocationType.MLP_POST_ACT,
                        layer_index=5,
                        tensor_indices=(0, 0),
                        pass_type=PassType.FORWARD,
                    )
                )
            ),
        ),
        multiple_top_k_derived_scalars_request_spec=MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id={
                GroupId.SINGLETON: [
                    DerivedScalarType.RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD,
                ]
            },
            # dsts for each group ID are assumed to have defined node_type,
            # all node_types assumed to be distinct within a group_id, and all group_ids to
            # contain the same set of node_types.
            token_index=None,
            top_and_bottom_k=100,
            pass_type=PassType.FORWARD,
        ),
    )
    single_response = await interactive_model.get_multiple_top_k_derived_scalars(single_request)

    # test layer requests
    single_request = MultipleTopKDerivedScalarsRequest(
        inference_request_spec=InferenceRequestSpec(
            prompt="Hello world",
            loss_fn_config=loss_fn_config,
        ),
        multiple_top_k_derived_scalars_request_spec=MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id={
                GroupId.SINGLETON: [
                    DerivedScalarType.RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD,
                ]
            },
            # dsts for each group ID are assumed to have defined node_type,
            # all node_types assumed to be distinct within a group_id, and all group_ids to
            # contain the same set of node_types.
            token_index=None,
            top_and_bottom_k=100,
            pass_type=PassType.FORWARD,
        ),
    )
    single_response = await interactive_model.get_multiple_top_k_derived_scalars(single_request)

    # test individual requests, and then a combined multi-request

    # the individual requests should share the same inference request spec
    inference_request_spec = InferenceRequestSpec(
        prompt="Hello world",
        loss_fn_config=loss_fn_config,
    )

    # test vocab token requests

    vocab_token_request_spec = MultipleTopKDerivedScalarsRequestSpec(
        dst_list_by_group_id={
            GroupId.LOGITS: [DerivedScalarType.LOGITS],
        },
        # dsts for each group ID are assumed to have defined node_type,
        # all node_types assumed to be distinct within a group_id, and all group_ids to
        # contain the same set of node_types.
        token_index=1,
        top_and_bottom_k=10,
        pass_type=PassType.FORWARD,
    )
    request = MultipleTopKDerivedScalarsRequest(
        inference_request_spec=inference_request_spec,
        multiple_top_k_derived_scalars_request_spec=vocab_token_request_spec,
    )
    vocab_token_response = await interactive_model.get_multiple_top_k_derived_scalars(request)
    vocab_token_response_data_in_single = (
        vocab_token_response.multiple_top_k_derived_scalars_response_data
    )
    print(vocab_token_response_data_in_single.activations_by_group_id[GroupId.LOGITS])

    unflattened_attention_dst_request_spec = MultipleTopKDerivedScalarsRequestSpec(
        dst_list_by_group_id=make_dst_list_by_group_id(
            group_ids=[
                GroupId.WRITE_NORM,
                GroupId.ACT_TIMES_GRAD,
                GroupId.DIRECT_WRITE_TO_GRAD,
            ],
            component_types=["mlp", "unflattened_attn"],
        ),
        # dsts for each group ID are assumed to have defined node_type,
        # all node_types assumed to be distinct within a group_id, and all group_ids to
        # contain the same set of node_types.
        token_index=None,
        top_and_bottom_k=10,
        pass_type=PassType.FORWARD,
    )

    # test requests with unflattened attention DSTs
    multi_top_k_request = MultipleTopKDerivedScalarsRequest(
        inference_request_spec=inference_request_spec,
        multiple_top_k_derived_scalars_request_spec=unflattened_attention_dst_request_spec,
    )

    unflattened_attention_response = await interactive_model.get_multiple_top_k_derived_scalars(
        multi_top_k_request
    )
    unflattened_attention_response_data_in_single = (
        unflattened_attention_response.multiple_top_k_derived_scalars_response_data
    )

    # test request for vocab tokens and flattened attention DSTs combined
    batched_request = BatchedRequest(
        inference_sub_requests=[
            InferenceSubRequest(
                inference_request_spec=inference_request_spec,
                processing_request_spec_by_name={
                    "vocab_token": vocab_token_request_spec,
                    "unflattened_attention": unflattened_attention_dst_request_spec,
                },
            ),
        ],
    )

    batched_response = await interactive_model.handle_batched_request(batched_request)
    assert len(batched_response.inference_sub_responses) == 1
    response = batched_response.inference_sub_responses[0]
    vocab_token_response_data = response.processing_response_data_by_name["vocab_token"]
    assert isinstance(vocab_token_response_data, MultipleTopKDerivedScalarsResponseData)

    assert (
        vocab_token_response_data_in_single.activations_by_group_id[GroupId.LOGITS]
        == vocab_token_response_data.activations_by_group_id[GroupId.LOGITS]
    )

    unflattened_attention_response_data = response.processing_response_data_by_name[
        "unflattened_attention"
    ]
    assert isinstance(
        unflattened_attention_response_data,
        MultipleTopKDerivedScalarsResponseData,
    )

    assert (
        unflattened_attention_response_data_in_single.node_indices
        == unflattened_attention_response_data.node_indices
    )
    for group_id in unflattened_attention_response_data_in_single.activations_by_group_id:
        assert (
            unflattened_attention_response_data_in_single.activations_by_group_id[group_id]
            == unflattened_attention_response_data.activations_by_group_id[group_id]
        )


async def test_top_tokens_timing(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context,
        standard_autoencoder_context,
    )

    loss_fn_config = LossFnConfig(
        name=LossFnName.LOGIT_DIFF,
        target_tokens=["!"],
        distractor_tokens=["."],
    )

    # the individual requests should share the same inference request spec
    inference_request_spec = InferenceRequestSpec(
        prompt="Hello world",
        loss_fn_config=loss_fn_config,
    )

    top_and_bottom_k = 10
    num_tokens = 30
    multi_top_k_request_spec = MultipleTopKDerivedScalarsRequestSpec(
        dst_list_by_group_id=make_dst_list_by_group_id(
            group_ids=[
                GroupId.DIRECT_WRITE_TO_GRAD,
            ],
            component_types=["mlp"],
        ),
        # dsts for each group ID are assumed to have defined node_type,
        # all node_types assumed to be distinct within a group_id, and all group_ids to
        # contain the same set of node_types.
        token_index=None,
        top_and_bottom_k=top_and_bottom_k,
        pass_type=PassType.FORWARD,
    )

    write_tokens_request_spec = ScoredTokensRequestSpec(
        token_scoring_type=TokenScoringType.UPVOTED_OUTPUT_TOKENS,
        num_tokens=num_tokens,
        depends_on_spec_name="multi_top_k",
    )

    read_tokens_request_spec = ScoredTokensRequestSpec(
        token_scoring_type=TokenScoringType.INPUT_TOKENS_THAT_UPVOTE_MLP,
        num_tokens=num_tokens,
        depends_on_spec_name="multi_top_k",
    )

    # test request for vocab tokens and flattened attention DSTs combined
    batched_request = BatchedRequest(
        inference_sub_requests=[
            InferenceSubRequest(
                inference_request_spec=inference_request_spec,
                processing_request_spec_by_name={
                    "multi_top_k": multi_top_k_request_spec,
                    "write_tokens": write_tokens_request_spec,
                    "read_tokens": read_tokens_request_spec,
                },
            ),
        ],
    )

    batched_response = await interactive_model.handle_batched_request(batched_request)


async def test_postprocessing(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context,
        standard_autoencoder_context,
    )

    loss_fn_config = LossFnConfig(
        name=LossFnName.LOGIT_DIFF,
        target_tokens=["!"],
        distractor_tokens=["."],
    )

    for component_type in ["mlp", "autoencoder"]:
        # test per-sequence-token attention DSTs in multi-request context
        multi_top_k_request = MultipleTopKDerivedScalarsRequest(
            inference_request_spec=InferenceRequestSpec(
                prompt="Hello world",
                loss_fn_config=loss_fn_config,
            ),
            multiple_top_k_derived_scalars_request_spec=MultipleTopKDerivedScalarsRequestSpec(
                dst_list_by_group_id=make_dst_list_by_group_id(
                    group_ids=[GroupId.ACTIVATION],
                    component_types=[component_type, "unflattened_attn"],
                ),
                # dsts for each group ID are assumed to have defined node_type,
                # all node_types assumed to be distinct within a group_id, and all group_ids to
                # contain the same set of node_types.
                token_index=None,
                top_and_bottom_k=10,
                pass_type=PassType.FORWARD,
            ),
        )

        multi_response = await interactive_model.get_multiple_top_k_derived_scalars(
            multi_top_k_request
        )

        postprocessor = TokenWriteConverter(
            model_context=standard_model_context,
            multi_autoencoder_context=standard_autoencoder_context,
        )

        # get example ds_index and activation
        node_indices = multi_response.multiple_top_k_derived_scalars_response_data.node_indices
        for node_index_index in [0, 1]:
            print()
            activation = torch.tensor(
                multi_response.multiple_top_k_derived_scalars_response_data.activations_by_group_id[
                    GroupId.ACTIVATION
                ][node_index_index],
                device=standard_model_context.device,
            )
            node_index = node_indices[node_index_index]
            print(f"{node_index.node_type=}")
            dst_list = multi_top_k_request.multiple_top_k_derived_scalars_request_spec.dst_list_by_group_id[
                GroupId.ACTIVATION
            ]
            matching_dsts = [dst for dst in dst_list if dst.node_type == node_index.node_type]
            assert len(matching_dsts) == 1, f"{matching_dsts=}, {node_index=}, {dst_list=}"
            dst = matching_dsts[0]

            if dst != DerivedScalarType.ATTN_QK_PROBS:
                # for non-attention activations, the activation preserves enough information to reconstruct the token space write vector

                ds_index = DerivedScalarIndex.from_node_index(
                    node_index,
                    dst,
                )

                token_write = postprocessor.postprocess_tensor(ds_index, activation)

                print(f"{token_write.shape=}")


async def test_all_layer_autoencoders_request(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context,
        standard_autoencoder_context,
    )

    loss_fn_config = LossFnConfig(
        name=LossFnName.LOGIT_DIFF,
        target_tokens=["!"],
        distractor_tokens=["."],
    )
    single_request = MultipleTopKDerivedScalarsRequest(
        inference_request_spec=InferenceRequestSpec(
            prompt="Hello world",
            loss_fn_config=loss_fn_config,
        ),
        multiple_top_k_derived_scalars_request_spec=MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id={
                GroupId.SINGLETON: [
                    DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
                ]
            },
            # dsts for each group ID are assumed to have defined node_type,
            # all node_types assumed to be distinct within a group_id, and all group_ids to
            # contain the same set of node_types.
            token_index=None,
            top_and_bottom_k=100,
            pass_type=PassType.FORWARD,
            dimensions_to_keep_for_intermediate_sum=[
                Dimension.SEQUENCE_TOKENS,
                # Dimension.AUTOENCODER_LATENTS,
            ],
        ),
    )
    single_response = await interactive_model.get_multiple_top_k_derived_scalars(single_request)

    dims_to_keep_by_ndims: dict[int, list[Dimension]] = {
        2: [
            Dimension.SEQUENCE_TOKENS,
            Dimension.AUTOENCODER_LATENTS,
        ],
        1: [
            Dimension.SEQUENCE_TOKENS,
        ],
        0: [],
    }

    single_request_by_ndims = {}
    for ndims in dims_to_keep_by_ndims.keys():
        single_request_by_ndims[ndims] = MultipleTopKDerivedScalarsRequest(
            inference_request_spec=InferenceRequestSpec(
                prompt="Hello world",
                loss_fn_config=loss_fn_config,
            ),
            multiple_top_k_derived_scalars_request_spec=MultipleTopKDerivedScalarsRequestSpec(
                dst_list_by_group_id={
                    GroupId.SINGLETON: [
                        DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
                    ]
                },
                # dsts for each group ID are assumed to have defined node_type,
                # all node_types assumed to be distinct within a group_id, and all group_ids to
                # contain the same set of node_types.
                token_index=None,
                top_and_bottom_k=100,
                pass_type=PassType.FORWARD,
                dimensions_to_keep_for_intermediate_sum=dims_to_keep_by_ndims[ndims],
            ),
        )
    single_response_by_ndims = {
        ndims: await interactive_model.get_multiple_top_k_derived_scalars(
            single_request_by_ndims[ndims]
        )
        for ndims in dims_to_keep_by_ndims.keys()
    }
    dtype_by_ndims = {
        ndims: {
            dst: type(
                single_response_by_ndims[
                    ndims
                ].multiple_top_k_derived_scalars_response_data.intermediate_sum_activations_by_dst_by_group_id[
                    GroupId.SINGLETON
                ][
                    dst
                ]
            )
            for dst in single_response_by_ndims[ndims]
            .multiple_top_k_derived_scalars_response_data.intermediate_sum_activations_by_dst_by_group_id[
                GroupId.SINGLETON
            ]
            .keys()
        }
        for ndims in dims_to_keep_by_ndims.keys()
    }
    for ndims in single_response_by_ndims.keys():
        single_response = single_response_by_ndims[ndims]
        sum_by_dst = single_response.multiple_top_k_derived_scalars_response_data.intermediate_sum_activations_by_dst_by_group_id[
            GroupId.SINGLETON
        ]
        for dst in sum_by_dst.values():
            if ndims == 2:
                assert isinstance(
                    dst, Tensor2D
                ), f"{dst=} for {dims_to_keep_by_ndims=}, {dtype_by_ndims=}"
            elif ndims == 1:
                assert isinstance(dst, Tensor1D)
            elif ndims == 0:
                assert isinstance(dst, Tensor0D)
            else:
                raise ValueError(f"Invalid ndims: {ndims}")


async def test_ablation_specs(standard_model_context: StandardModelContext) -> None:
    interactive_model = InteractiveModel.from_standard_model_context(standard_model_context)

    vocab_token_request_spec = MultipleTopKDerivedScalarsRequestSpec(
        dst_list_by_group_id={
            GroupId.LOGITS: [
                DerivedScalarType.LOGITS,
            ]
        },
        # dsts for each group ID are assumed to have defined node_type,
        # all node_types assumed to be distinct within a group_id, and all group_ids to
        # contain the same set of node_types.
        token_index=1,
        top_and_bottom_k=10,
        pass_type=PassType.FORWARD,
    )

    mlp_act_times_grad_request_spec = MultipleTopKDerivedScalarsRequestSpec(
        dst_list_by_group_id={
            GroupId.ACT_TIMES_GRAD: [
                DerivedScalarType.MLP_ACT_TIMES_GRAD,
            ]
        },
        token_index=None,
        top_and_bottom_k=100,
        pass_type=PassType.FORWARD,
    )

    attn_act_times_grad_request_spec = MultipleTopKDerivedScalarsRequestSpec(
        dst_list_by_group_id={
            GroupId.ACT_TIMES_GRAD: [
                DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD,
            ]
        },
        token_index=None,
        top_and_bottom_k=100,
        pass_type=PassType.FORWARD,
    )

    zero_ablation_specs = [
        AblationSpec(
            index=MirroredActivationIndex(
                layer_index=0,
                activation_location_type=ActivationLocationType.MLP_POST_ACT,
                tensor_indices=("All", 0),
                pass_type=PassType.FORWARD,
            ),
            value=0.0,
        )
    ]

    ablation_specs_by_ablation_setting = {
        "clean": None,
        "ablated": zero_ablation_specs,
    }

    logits_by_ablation_setting = {}
    mlp_act_times_grad_by_ablation_setting = {}
    attn_act_times_grad_by_ablation_setting = {}
    for ablation_setting, ablation_specs in ablation_specs_by_ablation_setting.items():
        print(f"Testing ablation setting: {ablation_setting}")
        request_spec = InferenceRequestSpec(
            prompt="Hello world",
            ablation_specs=ablation_specs,
            loss_fn_config=LossFnConfig(
                name=LossFnName.LOGIT_DIFF,
                target_tokens=["!"],
                distractor_tokens=["."],
            ),
        )
        request_spec2 = InferenceRequestSpec(
            prompt="Goodbye",
            ablation_specs=ablation_specs,
            loss_fn_config=LossFnConfig(
                name=LossFnName.LOGIT_DIFF,
                target_tokens=["?"],
                distractor_tokens=[","],
            ),
        )
        batched_request = BatchedRequest(
            inference_sub_requests=[
                InferenceSubRequest(
                    inference_request_spec=request_spec,
                    processing_request_spec_by_name={
                        "vocab_token": vocab_token_request_spec,
                        "mlp_act_times_grad": mlp_act_times_grad_request_spec,
                        "attn_act_times_grad": attn_act_times_grad_request_spec,
                    },
                ),
                InferenceSubRequest(
                    inference_request_spec=request_spec2,
                    processing_request_spec_by_name={
                        "vocab_token": vocab_token_request_spec,
                        "mlp_act_times_grad": mlp_act_times_grad_request_spec,
                        "attn_act_times_grad": attn_act_times_grad_request_spec,
                    },
                ),
            ],
        )

        batched_response = await interactive_model.handle_batched_request(batched_request)
        # We add two requests, so we should get two responses. For now, only check
        # the validity of the first response though.
        assert len(batched_response.inference_sub_responses) == 2
        response = batched_response.inference_sub_responses[0]
        top_k = response.processing_response_data_by_name["vocab_token"]
        assert isinstance(top_k, MultipleTopKDerivedScalarsResponseData)
        logits_by_ablation_setting[ablation_setting] = top_k.activations_by_group_id[GroupId.LOGITS]
        top_k = response.processing_response_data_by_name["mlp_act_times_grad"]
        assert isinstance(top_k, MultipleTopKDerivedScalarsResponseData)
        mlp_act_times_grad_by_ablation_setting[ablation_setting] = (
            top_k.activations_by_group_id[GroupId.ACT_TIMES_GRAD],
            top_k.node_indices,
        )
        top_k = response.processing_response_data_by_name["attn_act_times_grad"]
        assert isinstance(top_k, MultipleTopKDerivedScalarsResponseData)
        attn_act_times_grad_by_ablation_setting[ablation_setting] = (
            top_k.activations_by_group_id[GroupId.ACT_TIMES_GRAD],
            top_k.node_indices,
        )

    # Check if the output logits changed
    assert not np.array_equal(
        logits_by_ablation_setting["clean"],
        logits_by_ablation_setting["ablated"],
    )

    # Check if the activations changed
    assert not np.array_equal(
        mlp_act_times_grad_by_ablation_setting["clean"][0],
        mlp_act_times_grad_by_ablation_setting["ablated"][0],
    )

    # Check if the attn activations changed
    assert not np.array_equal(
        attn_act_times_grad_by_ablation_setting["clean"][0],
        attn_act_times_grad_by_ablation_setting["ablated"][0],
    )


async def test_batched_ablation_equivalent_to_single_ablation(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context,
        standard_autoencoder_context,
    )
    processing_request_spec_by_name: dict[str, ProcessingRequestSpec] = {
        "topKComponents": MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id={
                GroupId.WRITE_NORM: [
                    DerivedScalarType.UNFLATTENED_ATTN_WRITE_NORM,
                    DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM,
                ],
            },
            top_and_bottom_k=4,
            token_index=None,
        ),
    }

    req_a = InferenceSubRequest(
        inference_request_spec=InferenceRequestSpec(
            prompt="<|endoftext|>Paris, France. Ottawa,",
            loss_fn_config=LossFnConfig(
                name=LossFnName.LOGIT_DIFF,
                target_tokens=[" Canada"],
                distractor_tokens=[" Germany"],
            ),
            ablation_specs=[
                AblationSpec(
                    index=MirroredActivationIndex(
                        activation_location_type=ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
                        pass_type=PassType.FORWARD,
                        tensor_indices=(6, 2),
                        layer_index=8,
                    ),
                    value=0,
                ),
            ],
        ),
        processing_request_spec_by_name=processing_request_spec_by_name,
    )

    req_b = InferenceSubRequest(
        inference_request_spec=InferenceRequestSpec(
            prompt="<|endoftext|>Paris, France. Madrid,",
            loss_fn_config=LossFnConfig(
                name=LossFnName.LOGIT_DIFF,
                target_tokens=[" Canada"],
                distractor_tokens=[" Germany"],
            ),
            ablation_specs=[
                AblationSpec(
                    index=MirroredActivationIndex(
                        activation_location_type=ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
                        pass_type=PassType.FORWARD,
                        tensor_indices=(5, 1),
                        layer_index=7,
                    ),
                    value=0,
                ),
            ],
        ),
        processing_request_spec_by_name=processing_request_spec_by_name,
    )

    def get_acts_and_indices_from_response(
        response: BatchedResponse, sub_resp_index: int = 0
    ) -> tuple[list[float], list[MirroredNodeIndex]]:
        sub_response = response.inference_sub_responses[sub_resp_index]
        sub_response_data = sub_response.processing_response_data_by_name["topKComponents"]
        assert isinstance(
            sub_response_data, MultipleTopKDerivedScalarsResponseData
        ), f"{sub_response_data=}"
        assert len(list(sub_response_data.activations_by_group_id.keys())) == 1
        return (
            sub_response_data.activations_by_group_id[GroupId.WRITE_NORM],
            sub_response_data.node_indices,
        )

    resp_a = await interactive_model.handle_batched_request(
        BatchedRequest(inference_sub_requests=[req_a])
    )
    a_acts, a_indices = get_acts_and_indices_from_response(resp_a)
    resp_b = await interactive_model.handle_batched_request(
        BatchedRequest(inference_sub_requests=[req_b])
    )
    b_acts, b_indices = get_acts_and_indices_from_response(resp_b)
    resp_ab = await interactive_model.handle_batched_request(
        BatchedRequest(inference_sub_requests=[req_a, req_b])
    )
    batched_a_acts, batched_a_indices = get_acts_and_indices_from_response(
        resp_ab, sub_resp_index=0
    )
    batched_b_acts, batched_b_indices = get_acts_and_indices_from_response(
        resp_ab, sub_resp_index=1
    )

    assert set(a_indices + b_indices) == set(
        batched_a_indices
    ), f"{a_indices=}, {b_indices=}, {batched_a_indices=}"
    assert_common_acts_within_epsilon(a_acts, batched_a_acts, a_indices, batched_a_indices)

    assert set(a_indices + b_indices) == set(
        batched_b_indices
    ), f"{a_indices=}, {b_indices=}, {batched_b_indices=}"
    assert_common_acts_within_epsilon(b_acts, batched_b_acts, b_indices, batched_b_indices)


async def test_activation_grabbing(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    """Compute act-times-grad of all upstream nodes for a given node"""
    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context,
        standard_autoencoder_context,
    )

    # specify from what points the backward pass will be performed; either from the loss or
    # from an activation, running each backward pass one after another
    layer_index_for_grad = 6
    inference_kwargs_by_backward_pass_setting: dict[str, dict[str, Any]] = {
        "loss": {
            "loss_fn_config": LossFnConfig(
                name=LossFnName.LOGIT_DIFF,
                target_tokens=["!"],
                distractor_tokens=["."],
            ),
            "trace_config": None,
        },
        "activation": {
            "loss_fn_config": None,
            "trace_config": MirroredTraceConfig.from_trace_config(
                TraceConfig.from_activation_index(
                    ActivationIndex(
                        layer_index=layer_index_for_grad,
                        activation_location_type=ActivationLocationType.MLP_POST_ACT,
                        tensor_indices=(0, 1),
                        pass_type=PassType.FORWARD,
                    ),
                    detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST,
                )
            ),
        },
        "autoencoder_activation": {
            "loss_fn_config": None,
            "trace_config": MirroredTraceConfig.from_trace_config(
                TraceConfig.from_activation_index(
                    ActivationIndex(
                        layer_index=layer_index_for_grad,
                        activation_location_type=ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
                        tensor_indices=(0, 1),
                        pass_type=PassType.FORWARD,
                    ),
                    detach_layer_norm_scale=DETACH_LAYER_NORM_SCALE_FOR_TEST,
                )
            ),
        },
    }

    # For each node, compute the gradient from this node to all upstream nodes, and select the
    # top-k upstream nodes with the largest activation-times-gradient.
    grads_by_backward_pass_setting = {}  # more precisely "act-times-grad"
    for backward_pass_setting in [
        "loss",
        "activation",
        "autoencoder_activation",
    ]:
        inference_kwargs: dict[str, Any] = inference_kwargs_by_backward_pass_setting[
            backward_pass_setting
        ]
        grad_requiring_dst_request_spec = MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id=make_dst_list_by_group_id(
                group_ids=[
                    GroupId.ACT_TIMES_GRAD,
                ],
                component_types=[
                    "mlp",
                    "autoencoder",
                    "unflattened_attn",
                ],
            ),
            # dsts for each group ID are assumed to have defined node_type,
            # all node_types assumed to be distinct within a group_id, and all group_ids to
            # contain the same set of node_types.
            token_index=None,
            top_and_bottom_k=10,
            pass_type=PassType.FORWARD,
        )
        multiple_top_k_layers_request_spec = MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id={
                GroupId.SINGLETON: [
                    DerivedScalarType.RESID_POST_ATTN_PROJ_TO_FINAL_RESIDUAL_GRAD,
                ]
            },
            # dsts for each group ID are assumed to have defined node_type,
            # all node_types assumed to be distinct within a group_id, and all group_ids to
            # contain the same set of node_types.
            token_index=None,
            top_and_bottom_k=100,
            pass_type=PassType.FORWARD,
        )
        print(f"Testing backward pass setting: {backward_pass_setting}")
        batched_request = BatchedRequest(
            inference_sub_requests=[
                InferenceSubRequest(
                    inference_request_spec=InferenceRequestSpec(
                        prompt="Hello world",
                        **inference_kwargs,
                    ),
                    processing_request_spec_by_name={
                        "multi_top_k": grad_requiring_dst_request_spec,
                        "layers": multiple_top_k_layers_request_spec,
                    },
                ),
                InferenceSubRequest(
                    inference_request_spec=InferenceRequestSpec(
                        prompt="Goodbye",
                        **inference_kwargs,
                    ),
                    processing_request_spec_by_name={
                        "multi_top_k": grad_requiring_dst_request_spec,
                        "layers": multiple_top_k_layers_request_spec,
                    },
                ),
            ],
        )
        assert (
            batched_request.inference_sub_requests[0].inference_request_spec.trace_config
            is not None
            or batched_request.inference_sub_requests[0].inference_request_spec.loss_fn_config
            is not None
        )
        batched_response = await interactive_model.handle_batched_request(batched_request)
        # We add two requests, so we should get two responses. For now, only check
        # the validity of the first response though.
        assert len(batched_response.inference_sub_responses) == 2
        response = batched_response.inference_sub_responses[0]
        multi_top_k = response.processing_response_data_by_name["multi_top_k"]
        assert isinstance(multi_top_k, MultipleTopKDerivedScalarsResponseData)
        grads_by_backward_pass_setting[backward_pass_setting] = multi_top_k.activations_by_group_id[
            GroupId.ACT_TIMES_GRAD
        ]

    # Check that the values are different for different backward pass settings
    assert not np.array_equal(
        grads_by_backward_pass_setting["loss"],
        grads_by_backward_pass_setting["activation"],
    )
    assert not np.array_equal(
        grads_by_backward_pass_setting["activation"],
        grads_by_backward_pass_setting["autoencoder_activation"],
    )

    # Checking the extremal values of each backward pass setting
    # Note: will need to change if anything about the requests is changed
    top_value_by_backward_pass_setting = {
        "loss": 162.00540161132812,
        "activation": 0.19626690447330475,
        "autoencoder_activation": 0.44310837984085083,
    }
    for backward_pass_setting in [
        "loss",
        "activation",
        "autoencoder_activation",
    ]:
        grads = grads_by_backward_pass_setting[backward_pass_setting]
        assert math.isclose(
            grads[0], top_value_by_backward_pass_setting[backward_pass_setting], rel_tol=1e-4
        ), f"{grads[0]} != {top_value_by_backward_pass_setting[backward_pass_setting]}"


async def test_batched_activation_grabbing(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context,
        standard_autoencoder_context,
    )

    # Derived scalars
    batched_request = BatchedRequest(
        inference_sub_requests=[
            InferenceSubRequest(
                inference_request_spec=InferenceRequestSpec(
                    prompt="Hello world",
                ),
                processing_request_spec_by_name={
                    "foobar": DerivedScalarsRequestSpec(
                        dst=DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
                        layer_index=5,
                        activation_index=0,
                    ),
                },
            ),
            InferenceSubRequest(
                inference_request_spec=InferenceRequestSpec(
                    prompt="Goodbye",
                ),
                processing_request_spec_by_name={
                    "foobar2": DerivedScalarsRequestSpec(
                        dst=DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
                        layer_index=6,
                        activation_index=0,
                    ),
                },
            ),
        ],
    )
    batched_response = await interactive_model.handle_batched_request(batched_request)
    assert len(batched_response.inference_sub_responses) == 2

    # Verify that Pydantic can serialize and deserialize the request
    raw_request = batched_request.dict()
    reconstituted_request = pydantic.parse_obj_as(BatchedRequest, raw_request)
    assert isinstance(reconstituted_request, BatchedRequest)
    assert len(reconstituted_request.inference_sub_requests) == 2

    # Activations
    batched_request = BatchedRequest(
        inference_sub_requests=[
            InferenceSubRequest(
                inference_request_spec=InferenceRequestSpec(
                    prompt="hello world my name is bob",
                ),
                processing_request_spec_by_name={
                    "foobar": DerivedScalarsRequestSpec(
                        dst=AUTOENCODER_TEST_DST,
                        layer_index=5,
                        activation_index=100,
                    ),
                },
            ),
            InferenceSubRequest(
                inference_request_spec=InferenceRequestSpec(
                    prompt="goodbye world i hope your good",
                ),
                processing_request_spec_by_name={
                    "foobar2": DerivedScalarsRequestSpec(
                        dst=AUTOENCODER_TEST_DST,
                        layer_index=6,
                        activation_index=101,
                    ),
                },
            ),
        ],
    )
    batched_response = await interactive_model.handle_batched_request(batched_request)
    assert len(batched_response.inference_sub_responses) == 2


async def test_multi_topk_multiple_different_layer_indices(
    standard_model_context: StandardModelContext,
) -> None:
    interactive_model = InteractiveModel.from_standard_model_context(standard_model_context)

    # test per-sequence-token attention DSTs in multi-request context
    multi_top_k_request1 = MultipleTopKDerivedScalarsRequest(
        inference_request_spec=InferenceRequestSpec(
            prompt="Hello world",
        ),
        multiple_top_k_derived_scalars_request_spec=MultipleTopKDerivedScalarsRequestSpec(
            dst_list_by_group_id=make_dst_list_by_group_id(
                group_ids=[GroupId.WRITE_NORM],
                component_types=["mlp"],
            ),
            token_index=None,
            top_and_bottom_k=100,
            pass_type=PassType.FORWARD,
        ),
    )

    multi_response1 = await interactive_model.get_multiple_top_k_derived_scalars(
        multi_top_k_request1
    )

    multi_top_k_request2 = multi_top_k_request1.copy()
    multi_response2 = await interactive_model.get_multiple_top_k_derived_scalars(
        multi_top_k_request2
    )
    data1 = multi_response1.multiple_top_k_derived_scalars_response_data
    data2 = multi_response2.multiple_top_k_derived_scalars_response_data
    for i1, node_indices1 in enumerate(data1.node_indices):
        for i2, node_indices2 in enumerate(data2.node_indices):
            if node_indices1 == node_indices2:
                value1 = data1.activations_by_group_id[GroupId.WRITE_NORM][i1]
                value2 = data2.activations_by_group_id[GroupId.WRITE_NORM][i2]
                assert_acts_within_epsilon(
                    [value1],
                    [value2],
                    epsilon=0.1,
                )


def test_get_derived_scalars_for_prompt(
    standard_model_context: StandardModelContext,
) -> None:
    prompt = "<|endoftext|>1+1="
    target = "2"
    loss_fn_name = LossFnName.LOGIT_DIFF
    distractor = "1"
    loss_fn = maybe_construct_loss_fn_for_backward_pass(
        model_context=standard_model_context,
        config=LossFnConfig(
            name=loss_fn_name,
            target_tokens=[target],
            distractor_tokens=[distractor],
        ),
    )
    (
        ds_store,
        inference_and_token_data,
        _,  # raw activation store; not used by current test
    ) = get_derived_scalars_for_prompt(
        model_context=standard_model_context,
        dst_and_config_list=[
            (DerivedScalarType.MLP_POST_ACT, None),  # None -> default config
            (DerivedScalarType.ATTN_QK_PROBS, None),
        ],
        prompt=prompt,
        loss_fn_for_backward_pass=loss_fn,
    )


async def test_tdb_request_vs_transformer(
    standard_model_context: StandardModelContext,
) -> None:
    """
    Test that the top and bottom output token logits returned by a BatchedTdbRequest
    match those straightforwardly output by the Transformer. Also test that the top
    MLP activation returned by the BatchedTdbRequest matches the activation value stored
    using a hook on the Transformer forward pass directly.
    This test compares the simplest way to get outputs and activations from a Transformer, with the
    way that is used in TDB.
    """
    transformer = standard_model_context.get_or_create_model()

    prompt = "This is a test"
    input_token_ints = torch.tensor(
        standard_model_context.encode(prompt), device=standard_model_context.device
    ).unsqueeze(0)
    logits, _ = transformer(input_token_ints)
    # these are the logits for the next token in the sequence, directly from the Transformer
    logits = logits[0, -1, : standard_model_context.n_vocab]

    tdb_request = TdbRequestSpec(
        prompt=prompt,
        target_tokens=["."],
        distractor_tokens=["!"],
        component_type_for_mlp=ComponentTypeForMlp.NEURON,
        component_type_for_attention=ComponentTypeForAttention.ATTENTION_HEAD,
        top_and_bottom_k_for_node_table=5,
        hide_early_layers_when_ablating=False,
        node_ablations=None,
        upstream_node_to_trace=None,
        downstream_node_to_trace=None,
    )
    batched_tdb_request = BatchedTdbRequest(sub_requests=[tdb_request])

    interactive_model = InteractiveModel.from_standard_model_context(standard_model_context)

    batched_tdb_response = await interactive_model.handle_batched_tdb_request(batched_tdb_request)

    output_token_logits_response = batched_tdb_response.inference_sub_responses[
        0
    ].processing_response_data_by_name["topOutputTokenLogits"]
    assert isinstance(output_token_logits_response, MultipleTopKDerivedScalarsResponseData)
    logit_activations = output_token_logits_response.activations_by_group_id[GroupId.LOGITS]
    logit_indices = output_token_logits_response.node_indices
    # these are activations and NodeIndex objects associated with the top and bottom predicted
    # tokens for the next token in the sequence

    vocab_token_indices = [index.tensor_indices[1] for index in logit_indices]
    top_k_logits = len(vocab_token_indices) // 2
    # first half are largest
    assert logits.topk(top_k_logits).indices.tolist() == vocab_token_indices[:top_k_logits]
    assert logits.topk(top_k_logits).values.tolist() == logit_activations[:top_k_logits]
    # second half are smallest
    assert (
        logits.topk(top_k_logits, largest=False).indices.tolist()
        == vocab_token_indices[-top_k_logits:][::-1]
    )
    assert (
        logits.topk(top_k_logits, largest=False).values.tolist()
        == logit_activations[-top_k_logits:][::-1]
    )

    top_k_components_response = batched_tdb_response.inference_sub_responses[
        0
    ].processing_response_data_by_name["topKComponents"]
    assert isinstance(top_k_components_response, MultipleTopKDerivedScalarsResponseData)
    top_node_indices = top_k_components_response.node_indices
    top_activations = top_k_components_response.activations_by_group_id[GroupId.ACTIVATION]

    mlp_neuron_index, mlp_activation = next(
        (
            (node_index, activation)
            for node_index, activation in zip(top_node_indices, top_activations)
            if node_index.node_type == NodeType.MLP_NEURON
        ),
        (None, None),
    )  # get the first MLP neuron index returned by the TDB request
    assert mlp_neuron_index is not None

    hooks = TransformerHooks()
    stored_activation = {}

    def saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_activation["act"] = act[
            (0,) + mlp_neuron_index.tensor_indices
        ].item()  # 0 batch index
        return act  # store the activation value for the MLP neuron in question

    hooks.append_to_path(
        "mlp.post_act.fwd",
        AtLayers([assert_not_none(mlp_neuron_index.layer_index)]).append(saving_hook_fn),
    )

    transformer(input_token_ints, hooks=hooks)

    # the hook should have populated the dict with the MLP neuron's activation value
    assert stored_activation["act"] == mlp_activation


async def test_tdb_request_with_autoencoder_vs_transformer(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    """
    Test that the top MLP autoencoder latent activation returned by the BatchedTdbRequest
    matches the activation value stored using a hook on the Transformer forward pass directly.
    This test compares the simplest way to get activations from a Transformer, with the way that is
    used in TDB.
    """
    transformer = standard_model_context.get_or_create_model()

    prompt = "This is a test"
    input_token_ints = torch.tensor(
        standard_model_context.encode(prompt), device=standard_model_context.device
    ).unsqueeze(0)

    tdb_request = TdbRequestSpec(
        prompt=prompt,
        target_tokens=["."],
        distractor_tokens=["!"],
        component_type_for_mlp=ComponentTypeForMlp.AUTOENCODER_LATENT,
        component_type_for_attention=ComponentTypeForAttention.ATTENTION_HEAD,
        top_and_bottom_k_for_node_table=15,
        hide_early_layers_when_ablating=False,
        node_ablations=None,
        upstream_node_to_trace=None,
        downstream_node_to_trace=None,
    )
    batched_tdb_request = BatchedTdbRequest(sub_requests=[tdb_request])

    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context, standard_autoencoder_context
    )

    batched_tdb_response = await interactive_model.handle_batched_tdb_request(batched_tdb_request)

    top_k_components_response = batched_tdb_response.inference_sub_responses[
        0
    ].processing_response_data_by_name["topKComponents"]
    assert isinstance(top_k_components_response, MultipleTopKDerivedScalarsResponseData)
    top_node_indices = top_k_components_response.node_indices
    top_activations = top_k_components_response.activations_by_group_id[GroupId.ACTIVATION]
    ae_index, ae_activation = next(
        (
            (node_index, activation)
            for node_index, activation in zip(top_node_indices, top_activations)
            if node_index.node_type == NodeType.MLP_AUTOENCODER_LATENT
        ),
        (None, None),
    )  # get the first latent index returned by the TDB request
    assert ae_index is not None

    hooks = TransformerHooks()
    stored_activation = {}

    def saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_activation["act"] = act[
            (0,) + ae_index.tensor_indices[0:1]
        ]  # 0 batch index, keep token index from ae_index
        return act  # store the activation value for the latent in question

    hooks.append_to_path(
        "mlp.post_act.fwd",
        AtLayers([assert_not_none(ae_index.layer_index)]).append(saving_hook_fn),
    )

    transformer(input_token_ints, hooks=hooks)

    autoencoder = standard_autoencoder_context.get_autoencoder(ae_index.layer_index)

    # the hook should have populated the dict with the latent's activation value
    assert ae_activation is not None
    assert math.isclose(
        autoencoder.encode(stored_activation["act"])[ae_index.tensor_indices[-1]],
        ae_activation,
        rel_tol=1e-4,
    )


async def test_tdb_request_act_times_grad_from_intermediate_node(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    """
    Test that the top MLP autoencoder latent act * grad returned by the BatchedTdbRequest
    matches the activation value stored using a hook on the Transformer forward pass directly.

    This test compares the simplest way to get activations from a Transformer, with the
    way that is used in TDB.
    """
    transformer = standard_model_context.get_or_create_model()

    prompt = "This is a test"
    input_token_ints = torch.tensor(
        standard_model_context.encode(prompt), device=standard_model_context.device
    ).unsqueeze(0)

    node_index_for_bwd = MirroredNodeIndex(
        node_type=NodeType.MLP_NEURON,
        layer_index=5,
        tensor_indices=(1, 0),
        pass_type=PassType.FORWARD,
    )

    tdb_request = TdbRequestSpec(
        prompt=prompt,
        target_tokens=[],
        distractor_tokens=[],
        component_type_for_mlp=ComponentTypeForMlp.NEURON,
        component_type_for_attention=ComponentTypeForAttention.ATTENTION_HEAD,
        top_and_bottom_k_for_node_table=25,
        hide_early_layers_when_ablating=False,
        node_ablations=None,
        upstream_node_to_trace=NodeToTrace(
            node_index=node_index_for_bwd,
            attention_trace_type=None,
            downstream_trace_config=None,
        ),
        downstream_node_to_trace=None,
    )
    batched_tdb_request = BatchedTdbRequest(sub_requests=[tdb_request])

    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context, standard_autoencoder_context
    )

    batched_tdb_response = await interactive_model.handle_batched_tdb_request(batched_tdb_request)

    top_k_components_response = batched_tdb_response.inference_sub_responses[
        0
    ].processing_response_data_by_name["topKComponents"]
    assert isinstance(top_k_components_response, MultipleTopKDerivedScalarsResponseData)
    top_node_indices = top_k_components_response.node_indices
    top_act_times_grads = top_k_components_response.activations_by_group_id[GroupId.ACT_TIMES_GRAD]
    top_acts = top_k_components_response.activations_by_group_id[GroupId.ACTIVATION]

    node_index, tdb_act, tdb_act_times_grad = next(
        (
            (top_node_index, top_act, top_act_times_grad)
            for top_node_index, top_act, top_act_times_grad in zip(
                top_node_indices, top_acts, top_act_times_grads
            )
            if top_node_index.node_type == NodeType.MLP_NEURON
        ),
        (None, None, None),
    )  # get the first latent index returned by the TDB request
    assert node_index is not None

    hooks = TransformerHooks()
    stored_act = {}
    stored_grad = {}
    stored_act_for_bwd = {}

    def act_saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_act["value"] = act[
            (0,) + node_index.tensor_indices
        ]  # 0 batch index, keep token index from node_index
        return act  # store the activation value for the latent in question

    def grad_saving_hook_fn(grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_grad["value"] = grad[(0,) + node_index.tensor_indices]
        return grad

    def act_grabbing_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_act_for_bwd["value"] = act[(0,) + node_index_for_bwd.tensor_indices]
        return act

    def detach_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return act.detach()

    hooks = (
        hooks.append_to_path(
            "mlp.post_act.fwd",
            AtLayers([assert_not_none(node_index.layer_index)]).append(act_saving_hook_fn),
        )
        .append_to_path(
            "mlp.post_act.bwd",
            AtLayers([assert_not_none(node_index.layer_index)]).append(grad_saving_hook_fn),
        )
        .append_to_path(
            "resid.torso.ln_mlp.scale.fwd",
            AtLayers([assert_not_none(node_index_for_bwd.layer_index)]).append(detach_fn),
        )  # detach layer norm scale
        .append_to_path(
            "mlp.pre_act.fwd",
            AtLayers([assert_not_none(node_index_for_bwd.layer_index)]).append(
                act_grabbing_hook_fn
            ),
        )
    )

    transformer(input_token_ints, hooks=hooks)

    stored_act_for_bwd["value"].backward()

    assert isinstance(tdb_act, float)
    assert math.isclose(
        tdb_act,
        stored_act["value"].item(),
        rel_tol=1e-4,
    )

    assert isinstance(tdb_act_times_grad, float)
    assert math.isclose(
        tdb_act_times_grad,
        (stored_act["value"] * stored_grad["value"]).item(),
        rel_tol=1e-4,
    )


async def test_tdb_request_act_times_grad_from_intermediate_autoencoder_node(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    """
    Test that the top attention act * grad returned by the BatchedTdbRequest when tracing
    from an intermediate autoencoder latent matches the activation value stored using hooks
    on the Transformer forward pass directly.

    This test compares the simplest way to get activations from a Transformer, with the
    way that is used in TDB.
    """
    transformer = standard_model_context.get_or_create_model()

    prompt = "This is a test"
    input_token_ints = torch.tensor(
        standard_model_context.encode(prompt), device=standard_model_context.device
    ).unsqueeze(0)

    node_index_for_bwd = MirroredNodeIndex(
        node_type=NodeType.AUTOENCODER_LATENT,
        layer_index=5,
        tensor_indices=(1, 0),
        pass_type=PassType.FORWARD,
    )

    tdb_request = TdbRequestSpec(
        prompt=prompt,
        target_tokens=[],
        distractor_tokens=[],
        component_type_for_mlp=ComponentTypeForMlp.AUTOENCODER_LATENT,
        component_type_for_attention=ComponentTypeForAttention.ATTENTION_HEAD,
        top_and_bottom_k_for_node_table=25,
        hide_early_layers_when_ablating=False,
        node_ablations=None,
        upstream_node_to_trace=NodeToTrace(
            node_index=node_index_for_bwd,
            attention_trace_type=None,
            downstream_trace_config=None,
        ),
        downstream_node_to_trace=None,
    )
    batched_tdb_request = BatchedTdbRequest(sub_requests=[tdb_request])

    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context, standard_autoencoder_context
    )

    batched_tdb_response = await interactive_model.handle_batched_tdb_request(batched_tdb_request)

    top_k_components_response = batched_tdb_response.inference_sub_responses[
        0
    ].processing_response_data_by_name["topKComponents"]
    assert isinstance(top_k_components_response, MultipleTopKDerivedScalarsResponseData)
    top_node_indices = top_k_components_response.node_indices
    top_act_times_grads = top_k_components_response.activations_by_group_id[GroupId.ACT_TIMES_GRAD]
    top_acts = top_k_components_response.activations_by_group_id[GroupId.ACTIVATION]

    node_index, tdb_act, tdb_act_times_grad = next(
        (
            (top_node_index, top_act, top_act_times_grad)
            for top_node_index, top_act, top_act_times_grad in zip(
                top_node_indices, top_acts, top_act_times_grads
            )
            if top_node_index.node_type == NodeType.ATTENTION_HEAD and top_act_times_grad > 0.0
        ),
        (None, None, None),
    )  # get the first latent index returned by the TDB request
    assert node_index is not None

    hooks = TransformerHooks()
    stored_act = {}
    stored_grad = {}
    stored_act_for_bwd = {}

    def act_saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_act["value"] = act[
            (0,) + node_index.tensor_indices
        ]  # 0 batch index, keep token index from node_index
        return act  # store the activation value for the latent in question

    def grad_saving_hook_fn(grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_grad["value"] = grad[(0,) + node_index.tensor_indices]
        return grad

    autoencoder = standard_autoencoder_context.get_autoencoder(node_index_for_bwd.layer_index)

    def act_grabbing_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # apply autoencoder
        token_index_for_bwd, activation_index_for_bwd = node_index_for_bwd.tensor_indices
        assert isinstance(token_index_for_bwd, int)
        assert isinstance(activation_index_for_bwd, int)
        stored_act_for_bwd["value"] = autoencoder.encode_pre_act(
            act[0, token_index_for_bwd : token_index_for_bwd + 1]
        )[0, activation_index_for_bwd]
        return act

    def detach_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return act.detach()

    hooks = (
        hooks.append_to_path(
            "attn.qk_probs.fwd",
            AtLayers([assert_not_none(node_index.layer_index)]).append(act_saving_hook_fn),
        )
        .append_to_path(
            "attn.qk_probs.bwd",
            AtLayers([assert_not_none(node_index.layer_index)]).append(grad_saving_hook_fn),
        )
        .append_to_path(
            "resid.torso.ln_mlp.scale.fwd",
            AtLayers([assert_not_none(node_index_for_bwd.layer_index)]).append(detach_fn),
        )  # detach layer norm scale
        .append_to_path(
            "mlp.post_act.fwd",  # autoencoder is applied to mlp.post_act
            AtLayers([assert_not_none(node_index_for_bwd.layer_index)]).append(
                act_grabbing_hook_fn
            ),
        )
    )

    transformer(input_token_ints, hooks=hooks)

    stored_act_for_bwd["value"].backward()

    assert isinstance(tdb_act, float)
    assert math.isclose(
        tdb_act,
        stored_act["value"].item(),
        rel_tol=1e-4,
    )

    assert isinstance(tdb_act_times_grad, float)
    assert math.isclose(
        tdb_act_times_grad,
        (stored_act["value"] * stored_grad["value"]).item(),
        rel_tol=1e-4,
    )


async def test_tdb_request_act_times_grad_from_intermediate_autoencoder_node_trace_through_k(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    """
    Test that the top attention act * grad returned by the BatchedTdbRequest when tracing
    from an intermediate attention node through K matches the activation value stored using hooks
    on the Transformer forward pass directly.

    This test compares the simplest way to get activations from a Transformer, with the
    way that is used in TDB.
    """
    transformer = standard_model_context.get_or_create_model()

    prompt = "This is a test"
    input_token_ints = torch.tensor(
        standard_model_context.encode(prompt), device=standard_model_context.device
    ).unsqueeze(0)

    node_index_for_bwd = MirroredNodeIndex(
        node_type=NodeType.ATTENTION_HEAD,
        layer_index=5,
        tensor_indices=(1, 0, 0),
        pass_type=PassType.FORWARD,
    )

    tdb_request = TdbRequestSpec(
        prompt=prompt,
        target_tokens=[],
        distractor_tokens=[],
        component_type_for_mlp=ComponentTypeForMlp.AUTOENCODER_LATENT,
        component_type_for_attention=ComponentTypeForAttention.ATTENTION_HEAD,
        top_and_bottom_k_for_node_table=25,
        hide_early_layers_when_ablating=False,
        node_ablations=None,
        upstream_node_to_trace=NodeToTrace(
            node_index=node_index_for_bwd,
            attention_trace_type=AttentionTraceType.K,
            downstream_trace_config=None,
        ),
        downstream_node_to_trace=None,
    )
    batched_tdb_request = BatchedTdbRequest(sub_requests=[tdb_request])

    interactive_model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
        standard_model_context, standard_autoencoder_context
    )

    batched_tdb_response = await interactive_model.handle_batched_tdb_request(batched_tdb_request)

    top_k_components_response = batched_tdb_response.inference_sub_responses[
        0
    ].processing_response_data_by_name["topKComponents"]
    assert isinstance(top_k_components_response, MultipleTopKDerivedScalarsResponseData)
    top_node_indices = top_k_components_response.node_indices
    top_act_times_grads = top_k_components_response.activations_by_group_id[GroupId.ACT_TIMES_GRAD]
    top_acts = top_k_components_response.activations_by_group_id[GroupId.ACTIVATION]

    node_index, tdb_act, tdb_act_times_grad = next(
        (
            (top_node_index, top_act, top_act_times_grad)
            for top_node_index, top_act, top_act_times_grad in zip(
                top_node_indices, top_acts, top_act_times_grads
            )
            if top_node_index.node_type == NodeType.ATTENTION_HEAD and top_act_times_grad > 0.0
        ),
        (None, None, None),
    )  # get the first latent index returned by the TDB request
    assert node_index is not None

    hooks = TransformerHooks()
    stored_act = {}
    stored_grad = {}
    stored_act_for_bwd = {}

    def act_saving_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_act["value"] = act[
            (0,) + node_index.tensor_indices
        ]  # 0 batch index, keep token index from node_index
        return act  # store the activation value for the latent in question

    def grad_saving_hook_fn(grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        stored_grad["value"] = grad[(0,) + node_index.tensor_indices]
        return grad

    def act_grabbing_hook_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # apply autoencoder
        stored_act_for_bwd["value"] = act[(0,) + node_index_for_bwd.tensor_indices]
        return act

    def detach_fn(act: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return act.detach()

    hooks = (
        hooks.append_to_path(
            "attn.qk_probs.fwd",
            AtLayers([assert_not_none(node_index.layer_index)]).append(act_saving_hook_fn),
        )
        .append_to_path(
            "attn.qk_probs.bwd",
            AtLayers([assert_not_none(node_index.layer_index)]).append(grad_saving_hook_fn),
        )
        .append_to_path(
            "resid.torso.ln_attn.scale.fwd",
            AtLayers([assert_not_none(node_index_for_bwd.layer_index)]).append(detach_fn),
        )  # detach layer norm scale
        .append_to_path(
            "attn.q.fwd",
            AtLayers([assert_not_none(node_index_for_bwd.layer_index)]).append(detach_fn),
        )  # detach layer norm scale
        .append_to_path(
            "attn.qk_logits.fwd",
            AtLayers([assert_not_none(node_index_for_bwd.layer_index)]).append(
                act_grabbing_hook_fn
            ),
        )
    )

    transformer(input_token_ints, hooks=hooks)

    stored_act_for_bwd["value"].backward()

    assert isinstance(tdb_act, float)
    assert math.isclose(
        tdb_act,
        stored_act["value"].item(),
        rel_tol=1e-4,
    )

    assert isinstance(tdb_act_times_grad, float)
    assert math.isclose(
        tdb_act_times_grad,
        (stored_act["value"] * stored_grad["value"]).item(),
        rel_tol=1e-4,
    )


T = TypeVar("T")


def assert_not_none(value: T | None) -> T:
    assert value is not None
    return value
