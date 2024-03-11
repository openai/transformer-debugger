"""
This file contains a test checking that DerivedScalarStore values associated with the embedding are
nonzero in DerivedScalarStores using backward passes from early activations.
"""

import pytest
import torch
from _pytest.fixtures import FixtureRequest

from neuron_explainer.activation_server.derived_scalar_computation import (
    get_derived_scalars_for_prompt,
    maybe_construct_loss_fn_for_backward_pass,
)
from neuron_explainer.activation_server.requests_and_responses import (
    AblationSpec,
    GroupId,
    LossFnConfig,
    LossFnName,
)
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    ActivationIndex,
    MirroredActivationIndex,
    NodeIndex,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.postprocessing import TokenReadConverter
from neuron_explainer.activations.derived_scalars.scalar_deriver import DstConfig
from neuron_explainer.models.autoencoder_context import AutoencoderContext
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    NodeType,
    PassType,
)
from neuron_explainer.models.model_context import ModelContext, StandardModelContext


@pytest.fixture(params=["mlp", "attn"])
def grad_location(request: FixtureRequest) -> str:
    return request.param


def test_emb_dsts(
    standard_model_context: StandardModelContext,
    grad_location: str,
) -> None:
    dst_list_by_group_id = {
        GroupId.ACT_TIMES_GRAD: [
            DerivedScalarType.UNFLATTENED_ATTN_ACT_TIMES_GRAD,
            DerivedScalarType.TOKEN_ATTRIBUTION,
            DerivedScalarType.MLP_ACT_TIMES_GRAD,
        ],
        GroupId.DIRECT_WRITE_TO_GRAD: [
            DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.RESID_POST_EMBEDDING_PROJ_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD,
        ],
    }
    for group_id in [GroupId.ACT_TIMES_GRAD, GroupId.DIRECT_WRITE_TO_GRAD]:
        prompt = "This is a test"
        n_tokens = len(standard_model_context.encode(prompt))

        autoencoder_context = None

        if grad_location == "mlp":
            activation_index_for_grad = ActivationIndex(
                activation_location_type=ActivationLocationType.MLP_POST_ACT,
                layer_index=0,
                tensor_indices=(1, 0),
                pass_type=PassType.FORWARD,
            )
        else:
            assert grad_location == "attn"
            activation_index_for_grad = ActivationIndex(
                activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
                layer_index=0,
                tensor_indices=(3, 2, 0),
                pass_type=PassType.FORWARD,
            )

        dst_list = dst_list_by_group_id[group_id]

        dst_config = DstConfig(
            model_context=standard_model_context,
            autoencoder_context=autoencoder_context,
            trace_config=TraceConfig.from_activation_index(activation_index_for_grad),
        )
        dst_and_config_list: list[tuple[DerivedScalarType, DstConfig | None]] = [
            (dst, dst_config) for dst in dst_list
        ]

        current_ds_store, _, raw_store = get_derived_scalars_for_prompt(
            model_context=standard_model_context,
            autoencoder_context=autoencoder_context,
            prompt=prompt,
            trace_config=TraceConfig.from_activation_index(activation_index_for_grad),
            dst_and_config_list=dst_and_config_list,
        )

        vals, indices = current_ds_store.topk(50)
        emb_indices = [index for index in indices if index.dst.node_type == NodeType.LAYER]
        assert len(emb_indices) > 0
        zipped_vals_and_indices = list(zip(vals, indices))
        match grad_location:
            case "mlp":
                # at least one non-emb value should be nonzero
                assert any(
                    val != 0
                    for val, index in zipped_vals_and_indices
                    if index.dst.node_type != NodeType.LAYER
                )
            case "attn":
                # all non-emb values should be zero
                assert all(
                    val == 0
                    for val, index in zipped_vals_and_indices
                    if index.dst.node_type != NodeType.LAYER
                )
            case _:
                raise ValueError(f"Invalid grad_location: {grad_location}")


def _compute_top_token_ints_using_reconstituted_grad(
    model_context: ModelContext,
    autoencoder_context: AutoencoderContext | None,
    prompt: str,
    activation_location_type: ActivationLocationType,
    layer_index: int,
    tensor_indices: tuple[int, ...],
    top_tokens_to_check: int,
) -> list[int]:
    assert isinstance(model_context, StandardModelContext)

    activation_index_for_grad = ActivationIndex(
        activation_location_type=activation_location_type,
        layer_index=layer_index,
        tensor_indices=tensor_indices,
        pass_type=PassType.FORWARD,
    )

    dst_config = DstConfig(
        model_context=model_context,
        autoencoder_context=autoencoder_context,
        trace_config=TraceConfig.from_activation_index(activation_index_for_grad),
    )

    token_read_converter = TokenReadConverter(
        model_context=model_context, multi_autoencoder_context=autoencoder_context
    )

    dst_and_config_list: list[tuple[DerivedScalarType, DstConfig]] = [
        (
            DerivedScalarType.from_activation_location_type(
                activation_index_for_grad.activation_location_type
            ),
            dst_config,
        )
    ]

    post_dst_and_config_list = token_read_converter.get_input_dst_and_config_list(
        dst_and_config_list
    )

    ds_store, _, _ = get_derived_scalars_for_prompt(
        model_context=model_context,
        autoencoder_context=autoencoder_context,
        prompt=prompt,
        dst_and_config_list=dst_and_config_list + post_dst_and_config_list,  # type: ignore
    )

    node_index = NodeIndex.from_activation_index(activation_index_for_grad)

    token_vector = token_read_converter.postprocess(
        ds_store=ds_store,
        node_index=node_index,
    )
    topk = torch.topk(token_vector, top_tokens_to_check)

    top_token_ints_using_postprocess = topk.indices.tolist()

    return top_token_ints_using_postprocess


def _compute_top_and_bottom_token_ints_using_input_direction(
    model_context: ModelContext,
    autoencoder_context: AutoencoderContext | None,
    prompt: str,
    activation_location_type: ActivationLocationType,
    layer_index: int,
    tensor_indices: tuple[int, ...],
    top_tokens_to_check: int,
) -> tuple[list[int], list[int]]:
    assert isinstance(model_context, StandardModelContext)

    activation_index_for_fake_grad = ActivationIndex(
        activation_location_type=activation_location_type,
        layer_index=layer_index,
        tensor_indices=tensor_indices,
        pass_type=PassType.BACKWARD,
    )

    value_for_fake_grad = 1.0

    ablation_specs = [
        AblationSpec(
            index=MirroredActivationIndex.from_activation_index(activation_index_for_fake_grad),
            value=value_for_fake_grad,
        )
    ]

    dst_list = [DerivedScalarType.VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION]

    dst_config = DstConfig(
        model_context=model_context,
        autoencoder_context=autoencoder_context,
        activation_index_for_fake_grad=activation_index_for_fake_grad,
    )

    dst_and_config_list: list[tuple[DerivedScalarType, DstConfig | None]] = [
        (dst, dst_config) for dst in dst_list
    ]

    loss_fn_for_backward_pass = maybe_construct_loss_fn_for_backward_pass(
        model_context=model_context,
        config=LossFnConfig(
            name=LossFnName.ZERO,
        ),
    )

    current_ds_store, _, raw_store = get_derived_scalars_for_prompt(
        model_context=model_context,
        autoencoder_context=autoencoder_context,
        prompt=prompt,
        loss_fn_for_backward_pass=loss_fn_for_backward_pass,
        dst_and_config_list=dst_and_config_list,
        ablation_specs=ablation_specs,
    )

    top_tokens_to_check = 10

    # select just the first token
    values, indices = current_ds_store.apply_transform_fn_to_activations(lambda x: x[0:1]).topk(
        top_tokens_to_check
    )

    top_token_ints_using_fake_grad = [index.tensor_indices[1] for index in indices]

    values, indices = current_ds_store.apply_transform_fn_to_activations(lambda x: x[0:1]).topk(
        top_tokens_to_check, largest=False
    )

    bottom_token_ints_using_fake_grad = [index.tensor_indices[1] for index in indices]

    return top_token_ints_using_fake_grad, bottom_token_ints_using_fake_grad  # type: ignore


def test_vocab_token_write_to_mlp_similarity_and_autoencoder_smoke(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    # for token write to MLP: for each neuron, check that the top tokens according to
    # VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION are "similar" (>=1 element overlap) to the top tokens
    # according to TokenReadConverter or that the bottom tokens
    # according to VOCAB_TOKEN_WRITE_TO_INPUT_DIRECTION are "similar" to the top tokens
    # according to TokenReadConverter.
    # We expect this DST to fix a sign flip in TokenReadConverter
    # caused by the MLP gradient sometimes being negative and sometimes being positive.

    prompt = "This is a test"
    top_tokens_to_check = 10

    activation_location_type = ActivationLocationType.MLP_POST_ACT
    layer_index = 5
    activation_index = 0

    for activation_index in [0, 1, 2, 3]:
        tensor_indices = (1, activation_index)  # token index doesn't matter
        for activation_location_type in {
            ActivationLocationType.MLP_POST_ACT,
            ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
        }:
            (
                top_token_ints_using_input_direction,
                bottom_token_ints_using_input_direction,
            ) = _compute_top_and_bottom_token_ints_using_input_direction(
                model_context=standard_model_context,
                autoencoder_context=standard_autoencoder_context,
                prompt=prompt,
                activation_location_type=activation_location_type,
                layer_index=layer_index,
                tensor_indices=tensor_indices,
                top_tokens_to_check=top_tokens_to_check,
            )

            top_token_ints_using_postprocess = _compute_top_token_ints_using_reconstituted_grad(
                model_context=standard_model_context,
                autoencoder_context=standard_autoencoder_context,
                prompt=prompt,
                activation_location_type=activation_location_type,
                layer_index=layer_index,
                tensor_indices=tensor_indices,
                top_tokens_to_check=top_tokens_to_check,
            )

            if activation_location_type == ActivationLocationType.MLP_POST_ACT:
                top_overlap = len(
                    set(top_token_ints_using_input_direction)
                    & set(top_token_ints_using_postprocess)
                )
                bottom_overlap = len(
                    set(bottom_token_ints_using_input_direction)
                    & set(top_token_ints_using_postprocess)
                )
                if top_overlap >= 1:
                    assert bottom_overlap == 0
                else:
                    raise ValueError(
                        "Neither flipping nor unflipping worked\n"
                        f"Top tokens using input direction: {top_token_ints_using_input_direction}\n"
                        f"Bottom tokens using input direction: {bottom_token_ints_using_input_direction}\n"
                        f"Top tokens using postprocess: {top_token_ints_using_postprocess}\n"
                        f"{activation_location_type.value}, {layer_index}:{tensor_indices[-1]}\n"
                    )

                print("activation: ")
                print(activation_location_type.value, f"{layer_index}:{tensor_indices[-1]}")
                print("top tokens:")
                print(
                    standard_model_context.decode_token_list(top_token_ints_using_input_direction)
                )
