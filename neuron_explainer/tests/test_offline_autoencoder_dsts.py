import pytest
import torch

from neuron_explainer.activation_server.derived_scalar_computation import (
    get_derived_scalars_for_prompt,
)
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import (
    ActivationIndex,
    DerivedScalarIndex,
    TraceConfig,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import DstConfig
from neuron_explainer.activations.derived_scalars.tests.utils import (
    get_activation_shape,
    get_autoencoder_test_path,
)
from neuron_explainer.models.autoencoder_context import AutoencoderConfig, AutoencoderContext
from neuron_explainer.models.model_component_registry import ActivationLocationType, PassType
from neuron_explainer.models.model_context import StandardModelContext

gradient_dst_by_input_dst = {
    DerivedScalarType.MLP_POST_ACT: DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_MLP_POST_ACT_INPUT,
    DerivedScalarType.RESID_DELTA_MLP: DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT,
    DerivedScalarType.RESID_DELTA_ATTN: DerivedScalarType.AUTOENCODER_LATENT_GRAD_WRT_RESIDUAL_INPUT,
}


tested_dsts = [
    DerivedScalarType.ATTN_WRITE_TO_LATENT,
    DerivedScalarType.ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS,
    DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN,
]


@pytest.mark.parametrize("tested_dst", tested_dsts)
def test_backprop_to_attn_smoke(
    standard_model_context: StandardModelContext,
    tested_dst: DerivedScalarType,
) -> None:
    autoencoder_input_dst = DerivedScalarType.RESID_DELTA_ATTN
    latent_index_to_check = 3

    autoencoder_path = get_autoencoder_test_path(autoencoder_input_dst)
    prompt = "This is a test"
    n_tokens = len(standard_model_context.encode(prompt))

    layer_index = 5
    token_index_to_check = 1

    autoencoder_config = AutoencoderConfig(
        dst=autoencoder_input_dst,
        autoencoder_path_by_layer_index={
            layer_index: autoencoder_path for layer_index in range(standard_model_context.n_layers)
        },
    )
    autoencoder_context = AutoencoderContext(
        autoencoder_config=autoencoder_config, device=standard_model_context.device
    )

    # define which DST to derive
    dst_config_by_dst = {
        tested_dst: DstConfig(
            model_context=standard_model_context,
            autoencoder_config=autoencoder_config,
            trace_config=TraceConfig.from_activation_index(
                ActivationIndex(
                    activation_location_type=ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
                    layer_index=layer_index,
                    tensor_indices=("All", latent_index_to_check),
                    pass_type=PassType.FORWARD,
                )
            ),
            layer_indices=[layer_index],
        ),
    }
    dst_and_config_list: list[tuple[DerivedScalarType, DstConfig]] = list(dst_config_by_dst.items())

    # run the model and compute the requested derived scalars
    ds_store, _, raw_store = get_derived_scalars_for_prompt(
        model_context=standard_model_context,
        prompt=prompt,
        trace_config=TraceConfig.from_activation_index(
            ActivationIndex(
                activation_location_type=ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
                layer_index=layer_index,
                tensor_indices=(token_index_to_check, latent_index_to_check),
                pass_type=PassType.FORWARD,
            )
        ),
        dst_and_config_list=dst_and_config_list,  # type: ignore
        autoencoder_context=autoencoder_context,
    )

    # check that the derived scalar is not all zeros, and has the correct shape
    ds_index = DerivedScalarIndex(
        dst=tested_dst,
        pass_type=PassType.FORWARD,
        tensor_indices=(token_index_to_check,),
        layer_index=layer_index,
    )
    # The .to("cpu") shouldn't be necessary, but there's a strange Python crash we hit if we run on
    # a MacBook, which uses the "mps" device. This is a workaround.
    derived_scalar = ds_store[ds_index].to("cpu")
    assert (
        derived_scalar.shape
        == get_activation_shape(tested_dst, standard_model_context, n_tokens)[1:]
    )
    assert not torch.allclose(derived_scalar, torch.zeros_like(derived_scalar))


tested_dsts_batched = [
    DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED,
    DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED,
]


@pytest.mark.parametrize("dst_batched", tested_dsts_batched)
def test_attn_write_to_latent_dsts_batched(
    standard_model_context: StandardModelContext,
    dst_batched: DerivedScalarType,
) -> None:
    autoencoder_input_dst = DerivedScalarType.RESID_DELTA_ATTN
    latent_index_to_check = 3

    autoencoder_path = get_autoencoder_test_path(autoencoder_input_dst)
    prompt = "This is a test"

    dst_not_batched = {
        DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS_BATCHED: DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS,
        DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN_BATCHED: DerivedScalarType.ATTN_WRITE_TO_LATENT_PER_SEQUENCE_TOKEN,
    }[dst_batched]

    layer_index = 5
    token_index_to_check = 1

    autoencoder_config = AutoencoderConfig(
        dst=autoencoder_input_dst,
        autoencoder_path_by_layer_index={
            layer_index: autoencoder_path for layer_index in range(standard_model_context.n_layers)
        },
    )
    autoencoder_context = AutoencoderContext(
        autoencoder_config=autoencoder_config, device=standard_model_context.device
    )
    autoencoder_context.warmup()

    # define which DST to derive
    dst_config_by_dst = {
        dst_batched: DstConfig(
            model_context=standard_model_context,
            autoencoder_config=autoencoder_config,
            layer_indices=[layer_index],
        ),
        dst_not_batched: DstConfig(
            model_context=standard_model_context,
            autoencoder_config=autoencoder_config,
            trace_config=TraceConfig.from_activation_index(
                ActivationIndex(
                    activation_location_type=ActivationLocationType.ONLINE_AUTOENCODER_LATENT,
                    layer_index=layer_index,
                    pass_type=PassType.FORWARD,
                    tensor_indices=("All", latent_index_to_check),
                )
            ),
            layer_indices=[layer_index],
        ),
    }

    dst_and_config_list: list[tuple[DerivedScalarType, DstConfig]] = list(dst_config_by_dst.items())

    # run the model and compute the requested derived scalars
    ds_store, _, raw_store = get_derived_scalars_for_prompt(
        model_context=standard_model_context,
        prompt=prompt,
        loss_fn_for_backward_pass=None,
        dst_and_config_list=dst_and_config_list,  # type: ignore
        autoencoder_context=autoencoder_context,
    )

    # compare the two outputs
    ds_index_batched = DerivedScalarIndex(
        dst=dst_batched,
        pass_type=PassType.FORWARD,
        tensor_indices=(token_index_to_check,),
        layer_index=layer_index,
    )
    ds_index_not_batched = DerivedScalarIndex(
        dst=dst_not_batched,
        pass_type=PassType.FORWARD,
        tensor_indices=(token_index_to_check,),
        layer_index=layer_index,
    )
    derived_scalar_batched = ds_store[ds_index_batched]
    derived_scalar_not_batched = ds_store[ds_index_not_batched]
    autoencoder_context.warmup()
    assert derived_scalar_batched.shape[-1] == autoencoder_context.num_autoencoder_directions
    assert torch.allclose(
        derived_scalar_batched[..., latent_index_to_check], derived_scalar_not_batched
    )
