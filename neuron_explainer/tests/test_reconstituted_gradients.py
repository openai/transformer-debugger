import torch

from neuron_explainer.activation_server.derived_scalar_computation import (
    get_derived_scalars_for_prompt,
)
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.indexing import ActivationIndex, TraceConfig
from neuron_explainer.activations.derived_scalars.locations import (
    get_previous_residual_dst_for_node_type,
)
from neuron_explainer.activations.derived_scalars.make_scalar_derivers import make_scalar_deriver
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    ActivationsAndMetadata,
    DstConfig,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    Dimension,
    PassType,
)
from neuron_explainer.models.model_context import StandardModelContext


def run_smoke_test(
    gpt2_small_model_context: StandardModelContext,
    n_tokens: int,
    activation_indices_for_grad: list[ActivationIndex],
    dst_and_shape: tuple[DerivedScalarType, tuple[int, ...]],
    test_name: str,
) -> None:
    print(f"Running smoke test for {test_name}")
    for activation_index_for_grad in activation_indices_for_grad:
        trace_config = TraceConfig.from_activation_index(activation_index_for_grad)
        dst_config = DstConfig(
            model_context=gpt2_small_model_context,
            trace_config=trace_config,
        )
        assert dst_config.trace_config is not None
        dst, shape = dst_and_shape
        # create scalar deriver
        scalar_deriver = make_scalar_deriver(dst, dst_config)

        residual_dst = get_previous_residual_dst_for_node_type(
            dst_config.trace_config.node_type, autoencoder_dst=None
        )

        print(f"{residual_dst=}")

        # create fake dataset of activations
        activations_and_metadata_tuple = create_fake_dataset_of_activations(
            dst,
            residual_dst,
            n_tokens,
            gpt2_small_model_context,
        )

        # calculate derived scalar
        new_activations_and_metadata = (
            scalar_deriver.activations_and_metadata_calculate_derived_scalar_fn(
                activations_and_metadata_tuple, PassType.FORWARD
            )
        )
        layer_indices = list(new_activations_and_metadata.activations_by_layer_index.keys())
        assert (
            new_activations_and_metadata.activations_by_layer_index[layer_indices[0]].shape == shape
        ), (
            new_activations_and_metadata.activations_by_layer_index[layer_indices[0]].shape,
            shape,
        )
        assert new_activations_and_metadata.dst == dst


def create_fake_dataset_of_activations(
    dst: DerivedScalarType,
    residual_dst: DerivedScalarType,
    n_tokens: int,
    model_context: StandardModelContext,
) -> tuple[ActivationsAndMetadata, ...]:
    layer_indices = list(range(model_context.n_layers))
    resid_activations_and_metadata = (
        ActivationsAndMetadata(
            activations_by_layer_index={
                layer_index: torch.randn(
                    (n_tokens, model_context.n_residual_stream_channels),
                    device=model_context.device,
                )
                for layer_index in layer_indices
            },
            pass_type=PassType.FORWARD,
            dst=residual_dst,
        ),
    )
    if dst == DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD:
        activations_and_metadata_tuple: tuple[ActivationsAndMetadata, ...] = (
            ActivationsAndMetadata(
                activations_by_layer_index={
                    layer_index: torch.randn(
                        (
                            n_tokens,
                            n_tokens,
                            model_context.n_attention_heads,
                            model_context.get_dim_size(Dimension.VALUE_CHANNELS),
                        ),
                        device=model_context.device,
                    )
                    for layer_index in layer_indices
                },
                pass_type=PassType.FORWARD,
                dst=DerivedScalarType.ATTN_WEIGHTED_VALUE,
            ),
        ) + resid_activations_and_metadata
    elif dst == DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD:
        activations_and_metadata_tuple = (
            ActivationsAndMetadata(
                activations_by_layer_index={
                    layer_index: torch.randn(
                        (n_tokens, model_context.n_neurons),
                        device=model_context.device,
                    )
                    for layer_index in layer_indices
                },
                pass_type=PassType.FORWARD,
                dst=DerivedScalarType.MLP_POST_ACT,
            ),
        ) + resid_activations_and_metadata
    else:
        raise ValueError("Invalid activation location type")

    return activations_and_metadata_tuple


def test_mlp_write_to_final_activation_residual_grad_smoke(
    standard_model_context: StandardModelContext,
) -> None:
    n_tokens = 10
    n_neurons = 3072
    run_smoke_test(
        gpt2_small_model_context=standard_model_context,
        n_tokens=n_tokens,
        activation_indices_for_grad=[
            ActivationIndex(
                activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
                layer_index=5,
                tensor_indices=(0, 0, 0),
                pass_type=PassType.FORWARD,
            ),
            ActivationIndex(
                activation_location_type=ActivationLocationType.MLP_POST_ACT,
                layer_index=5,
                tensor_indices=(0, 0),
                pass_type=PassType.FORWARD,
            ),
        ],
        dst_and_shape=(
            DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
            (n_tokens, n_neurons),
        ),
        test_name="MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD",
    )


def test_attn_write_to_final_activation_residual_grad_smoke(
    standard_model_context: StandardModelContext,
) -> None:
    n_tokens = 10
    n_attention_heads = 12
    run_smoke_test(
        gpt2_small_model_context=standard_model_context,
        n_tokens=n_tokens,
        activation_indices_for_grad=[
            ActivationIndex(
                activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
                layer_index=5,
                tensor_indices=(0, 0, 0),
                pass_type=PassType.FORWARD,
            ),
            ActivationIndex(
                activation_location_type=ActivationLocationType.MLP_POST_ACT,
                layer_index=5,
                tensor_indices=(0, 0),
                pass_type=PassType.FORWARD,
            ),
        ],
        dst_and_shape=(
            DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
            (n_tokens, n_tokens, n_attention_heads),
        ),
        test_name="ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD",
    )


def test_write_to_final_activation_residual_grad_equality(
    standard_model_context: StandardModelContext,
) -> None:
    prompt = "This is a test"

    dst_list = [
        DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
        DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
        DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD,
        DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD,
    ]

    activation_index_for_grad_list = [
        ActivationIndex(
            activation_location_type=ActivationLocationType.ATTN_QK_PROBS,
            layer_index=5,
            tensor_indices=(
                1,
                1,
                0,
            ),  # (0, 0, 0) is constrained to be 1 always, so gradient is zero
            pass_type=PassType.FORWARD,
        ),
        ActivationIndex(
            activation_location_type=ActivationLocationType.MLP_POST_ACT,
            layer_index=5,
            tensor_indices=(1, 0),
            pass_type=PassType.FORWARD,
        ),
    ]

    for activation_index_for_grad in activation_index_for_grad_list:
        trace_config = TraceConfig.from_activation_index(activation_index_for_grad)
        dst_config = DstConfig(
            model_context=standard_model_context,
            trace_config=trace_config,
        )
        dst_and_config_list = [(dst, dst_config) for dst in dst_list]

        ds_store, _, raw_store = get_derived_scalars_for_prompt(
            model_context=standard_model_context,
            prompt=prompt,
            trace_config=trace_config,
            dst_and_config_list=dst_and_config_list,  # type: ignore
        )

        if (
            activation_index_for_grad.activation_location_type
            == ActivationLocationType.ATTN_QK_PROBS
        ):
            layer_indices = [3, 4]
            resid_activation_location_type = ActivationLocationType.RESID_POST_MLP
        else:
            assert (
                activation_index_for_grad.activation_location_type
                == ActivationLocationType.MLP_POST_ACT
            )
            layer_indices = [3, 4, 5]
            resid_activation_location_type = ActivationLocationType.RESID_POST_ATTN

        for layer_index in layer_indices:
            print(
                layer_index,
                raw_store.get_activations_and_metadata(
                    activation_location_type=resid_activation_location_type,
                    pass_type=PassType.BACKWARD,
                ).activations_by_layer_index[layer_index][:, 0],
            )
        activation_dst_to_vanilla_dst = {
            DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
            DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: DerivedScalarType.MLP_WRITE_TO_FINAL_RESIDUAL_GRAD,
        }
        activation_dst_to_slicer: dict[DerivedScalarType, tuple[slice | int, ...]] = {
            DerivedScalarType.UNFLATTENED_ATTN_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
                slice(None),
                slice(None),
                0,
            ),
            DerivedScalarType.MLP_WRITE_TO_FINAL_ACTIVATION_RESIDUAL_GRAD: (
                slice(None),
                slice(None, 5),
            ),
        }
        for activation_dst in activation_dst_to_vanilla_dst.keys():
            pass_type = PassType.FORWARD
            layer_index_to_print = 4
            if activation_dst in dst_list:
                vanilla_dst = activation_dst_to_vanilla_dst[activation_dst]
                slicer = activation_dst_to_slicer[activation_dst]
                print(
                    activation_dst,
                    activation_index_for_grad.activation_location_type,
                    ds_store.activations_and_metadata_by_dst_and_pass_type[
                        (
                            activation_dst,
                            pass_type,
                        )
                    ].activations_by_layer_index[layer_index_to_print][slicer],
                    ds_store.activations_and_metadata_by_dst_and_pass_type[
                        (
                            activation_dst,
                            pass_type,
                        )
                    ]
                    .activations_by_layer_index[layer_index_to_print]
                    .shape,
                    ds_store.activations_and_metadata_by_dst_and_pass_type[
                        (vanilla_dst, pass_type)
                    ].activations_by_layer_index[layer_index_to_print][slicer],
                    ds_store.activations_and_metadata_by_dst_and_pass_type[(vanilla_dst, pass_type)]
                    .activations_by_layer_index[layer_index_to_print]
                    .shape,
                )
                assert (
                    ds_store.activations_and_metadata_by_dst_and_pass_type[
                        (
                            activation_dst,
                            pass_type,
                        )
                    ]
                    == ds_store.activations_and_metadata_by_dst_and_pass_type[
                        (vanilla_dst, pass_type)
                    ]
                )
