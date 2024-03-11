from typing import Callable

import torch

from neuron_explainer.activation_server.derived_scalar_computation import (
    DerivedScalarComputationParams,
    compute_derived_scalar_groups_for_input_token_ints,
)
from neuron_explainer.activation_server.requests_and_responses import InferenceData
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.derived_scalar_store import DerivedScalarStore
from neuron_explainer.activations.derived_scalars.indexing import AblationSpec, TraceConfig
from neuron_explainer.activations.derived_scalars.make_scalar_derivers import make_scalar_deriver
from neuron_explainer.activations.derived_scalars.multi_group import (
    MultiGroupDerivedScalarStore,
    MultiGroupScalarDerivers,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    ActivationsAndMetadata,
    DstConfig,
    ScalarDeriver,
)
from neuron_explainer.models.autoencoder_context import AutoencoderContext, MultiAutoencoderContext
from neuron_explainer.models.model_component_registry import PassType
from neuron_explainer.models.model_context import StandardModelContext
from neuron_explainer.tests.conftest import AUTOENCODER_TEST_DST


def validate_activations_and_metadata(
    activations_and_metadata: ActivationsAndMetadata,
    dst: DerivedScalarType,
    pass_type: PassType,
    input_token_ints: list[list[int]],
) -> None:
    assert activations_and_metadata.dst == dst
    assert activations_and_metadata.pass_type == pass_type
    for layer_index in activations_and_metadata.activations_by_layer_index.keys():
        assert activations_and_metadata.activations_by_layer_index[layer_index].shape[0] == len(
            input_token_ints[0]
        )


def compute_derived_scalars_for_input_token_ints(
    model_context: StandardModelContext,
    input_token_ints: list[list[int]],
    scalar_derivers: list[ScalarDeriver],
    loss_fn_for_backward_pass: Callable[[torch.Tensor], torch.Tensor] | None,
    ablation_specs: list[AblationSpec] | None,
    autoencoder_context: MultiAutoencoderContext | AutoencoderContext | None,
    trace_config: TraceConfig | None,
) -> tuple[DerivedScalarStore, InferenceData]:
    """This function runs a forward pass on the given input tokens, with hooks added to the transformer to
    extract the activations needed to compute the scalars in scalar_derivers. It then returns a
    DerivedScalarStore containing the derived scalars for each token in the input."""
    multi_autoencoder_context = MultiAutoencoderContext.from_context_or_multi_context(
        autoencoder_context
    )
    multi_group_scalar_derivers_by_processing_step = {
        # "dummy" is a placeholder processing step name
        "dummy": MultiGroupScalarDerivers.from_scalar_derivers(scalar_derivers)
    }
    assert len(input_token_ints) == 1
    batched_ds_computation_params = [
        DerivedScalarComputationParams(
            input_token_ints=input_token_ints[0],
            multi_group_scalar_derivers_by_processing_step=multi_group_scalar_derivers_by_processing_step,
            device_for_raw_activations=model_context.device,
            loss_fn_for_backward_pass=loss_fn_for_backward_pass,
            ablation_specs=ablation_specs,
            trace_config=trace_config,
        )
    ]
    batched_multi_group_ds_store_by_processing_step: list[dict[str, MultiGroupDerivedScalarStore]]
    (
        batched_multi_group_ds_store_by_processing_step,
        batched_inference_data,
        _,
    ) = compute_derived_scalar_groups_for_input_token_ints(
        model_context=model_context,
        multi_autoencoder_context=multi_autoencoder_context,
        batched_ds_computation_params=batched_ds_computation_params,
    )
    return (
        batched_multi_group_ds_store_by_processing_step[0]["dummy"].to_single_ds_store(),
        batched_inference_data[0],
    )


def test_online_autoencoder_latent(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    # SANITY CHECK ONLINE AUTOENCODER LATENT DST OUTPUTS
    autoencoder_config = standard_autoencoder_context.autoencoder_config
    dst_config = DstConfig(
        model_context=standard_model_context,
        autoencoder_config=autoencoder_config,
    )
    online_latent_scalar_deriver = make_scalar_deriver(
        DerivedScalarType.ONLINE_AUTOENCODER_LATENT, dst_config
    )

    mlp_layer_index = 5
    dst_config = DstConfig(
        model_context=standard_model_context,
    )
    mlp_scalar_deriver = make_scalar_deriver(
        AUTOENCODER_TEST_DST,
        dst_config,
    )

    scalar_derivers = [online_latent_scalar_deriver, mlp_scalar_deriver]

    input_token_ints = [[1, 2, 3, 4, 5]]
    ds_store_with_autoencoder, loss = compute_derived_scalars_for_input_token_ints(
        standard_model_context,
        input_token_ints,
        scalar_derivers,
        loss_fn_for_backward_pass=None,
        ablation_specs=None,
        autoencoder_context=standard_autoencoder_context,
        trace_config=None,
    )

    activations_and_metadata_by_dst_and_pass_type = (
        ds_store_with_autoencoder.activations_and_metadata_by_dst_and_pass_type
    )
    assert len(activations_and_metadata_by_dst_and_pass_type) == 2
    dst_and_pass_type = (DerivedScalarType.ONLINE_AUTOENCODER_LATENT, PassType.FORWARD)

    activations_and_metadata = activations_and_metadata_by_dst_and_pass_type[dst_and_pass_type]

    validate_activations_and_metadata(
        activations_and_metadata,
        DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
        PassType.FORWARD,
        input_token_ints,
    )

    dst_and_pass_type = (AUTOENCODER_TEST_DST, PassType.FORWARD)
    mlp_activations_and_metadata_with_autoencoder = activations_and_metadata_by_dst_and_pass_type[
        dst_and_pass_type
    ]

    validate_activations_and_metadata(
        mlp_activations_and_metadata_with_autoencoder,
        AUTOENCODER_TEST_DST,
        PassType.FORWARD,
        input_token_ints,
    )

    # CHECK ONLINE AUTOENCODER HOOK DOES NOT AFFECT DOWNSTREAM ACTIVATIONS, BY COMPARING TO CASE WHERE HOOK
    # IS NOT INCLUDED
    scalar_derivers = [mlp_scalar_deriver]

    input_token_ints = [[1, 2, 3, 4, 5]]
    (
        ds_store_without_autoencoder,
        loss,
    ) = compute_derived_scalars_for_input_token_ints(
        standard_model_context,
        input_token_ints,
        scalar_derivers,
        loss_fn_for_backward_pass=None,
        ablation_specs=None,
        autoencoder_context=None,
        trace_config=None,
    )

    activations_and_metadata_by_dst_and_pass_type_without_autoencoder = (
        ds_store_without_autoencoder.activations_and_metadata_by_dst_and_pass_type
    )
    dst_and_pass_type = (AUTOENCODER_TEST_DST, PassType.FORWARD)
    mlp_activations_and_metadata_without_autoencoder = (
        activations_and_metadata_by_dst_and_pass_type_without_autoencoder[dst_and_pass_type]
    )

    validate_activations_and_metadata(
        mlp_activations_and_metadata_without_autoencoder,
        AUTOENCODER_TEST_DST,
        PassType.FORWARD,
        input_token_ints,
    )

    assert (
        mlp_activations_and_metadata_with_autoencoder.activations_by_layer_index[
            mlp_layer_index
        ].shape
        == mlp_activations_and_metadata_without_autoencoder.activations_by_layer_index[
            mlp_layer_index
        ].shape
    )
    assert torch.allclose(
        mlp_activations_and_metadata_with_autoencoder.activations_by_layer_index[mlp_layer_index],
        mlp_activations_and_metadata_without_autoencoder.activations_by_layer_index[
            mlp_layer_index
        ],
        rtol=1e-4,
        atol=1e-4,
    ), (
        mlp_activations_and_metadata_with_autoencoder.activations_by_layer_index[
            mlp_layer_index
        ].flatten()[:10],
        mlp_activations_and_metadata_without_autoencoder.activations_by_layer_index[
            mlp_layer_index
        ].flatten()[:10],
    )


def test_online_equal_to_offline(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    # CHECK THAT ONLINE AND OFFLINE AUTOENCODER LATENT DSTS RESULT IN THE SAME ACTIVATIONS
    dst_config = DstConfig(
        model_context=standard_model_context,
        autoencoder_config=standard_autoencoder_context.autoencoder_config,
    )
    online_latent_scalar_deriver = make_scalar_deriver(
        DerivedScalarType.ONLINE_AUTOENCODER_LATENT, dst_config
    )
    offline_latent_scalar_deriver = make_scalar_deriver(
        DerivedScalarType.AUTOENCODER_LATENT, dst_config
    )

    scalar_derivers = [online_latent_scalar_deriver, offline_latent_scalar_deriver]

    input_token_ints = [[1, 2, 3, 4, 5]]
    ds_store, loss = compute_derived_scalars_for_input_token_ints(
        standard_model_context,
        input_token_ints,
        scalar_derivers,
        loss_fn_for_backward_pass=None,
        ablation_specs=None,
        autoencoder_context=standard_autoencoder_context,
        trace_config=None,
    )

    activations_and_metadata_by_dst_and_pass_type = (
        ds_store.activations_and_metadata_by_dst_and_pass_type
    )
    assert len(activations_and_metadata_by_dst_and_pass_type) == 2

    dst_and_pass_type = (DerivedScalarType.ONLINE_AUTOENCODER_LATENT, PassType.FORWARD)
    online_activations_and_metadata = activations_and_metadata_by_dst_and_pass_type[
        dst_and_pass_type
    ]

    validate_activations_and_metadata(
        online_activations_and_metadata,
        DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
        PassType.FORWARD,
        input_token_ints,
    )

    dst_and_pass_type = (DerivedScalarType.AUTOENCODER_LATENT, PassType.FORWARD)
    offline_activations_and_metadata = activations_and_metadata_by_dst_and_pass_type[
        dst_and_pass_type
    ]
    validate_activations_and_metadata(
        offline_activations_and_metadata,
        DerivedScalarType.AUTOENCODER_LATENT,
        PassType.FORWARD,
        input_token_ints,
    )
    # check that online and offline activations are the same
    for layer_index in range(standard_model_context.n_layers):
        assert torch.allclose(
            online_activations_and_metadata.activations_by_layer_index[layer_index],
            offline_activations_and_metadata.activations_by_layer_index[layer_index],
        )


def test_autoencoder_dsts_together(
    standard_model_context: StandardModelContext,
    standard_autoencoder_context: AutoencoderContext,
) -> None:
    # test that autoencoder act * grad works together with other autoencoder dsts
    dst_config = DstConfig(
        model_context=standard_model_context,
        autoencoder_config=standard_autoencoder_context.autoencoder_config,
    )
    dsts = [
        DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
        DerivedScalarType.ONLINE_AUTOENCODER_WRITE_NORM,
        DerivedScalarType.ONLINE_AUTOENCODER_ACT_TIMES_GRAD,
        DerivedScalarType.ONLINE_AUTOENCODER_WRITE_TO_FINAL_RESIDUAL_GRAD,
        DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR,
        DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_ACT_TIMES_GRAD,
        DerivedScalarType.ONLINE_MLP_AUTOENCODER_ERROR_WRITE_TO_FINAL_RESIDUAL_GRAD,
    ]

    scalar_derivers = [make_scalar_deriver(dst, dst_config) for dst in dsts]

    def loss_fn_for_backward_pass(output_logits: torch.Tensor) -> torch.Tensor:
        assert output_logits.ndim == 3
        nbatch, ntoken, nlogit = output_logits.shape
        assert nbatch == 1
        target_vocab_token_ints = [0]
        anti_target_vocab_token_ints = [1]
        return (
            output_logits[:, -1, target_vocab_token_ints].mean(-1)
            - output_logits[:, -1, anti_target_vocab_token_ints].mean(-1)
        ).mean()  # difference between average logits for target and anti-target tokens

    input_token_ints = [[1, 2, 3, 4, 5]]
    ds_store, loss = compute_derived_scalars_for_input_token_ints(
        standard_model_context,
        input_token_ints,
        scalar_derivers,
        loss_fn_for_backward_pass=loss_fn_for_backward_pass,
        ablation_specs=None,
        autoencoder_context=standard_autoencoder_context,
        trace_config=None,
    )
