import torch

from neuron_explainer.activations.derived_scalars.derived_scalar_store import (
    DerivedScalarIndex,
    DerivedScalarStore,
    RawActivationStore,
)
from neuron_explainer.activations.derived_scalars.make_scalar_derivers import make_scalar_deriver
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    ActivationsAndMetadata,
    DerivedScalarType,
    DstConfig,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationTypeAndPassType,
    Dimension,
    LayerIndex,
    PassType,
)
from neuron_explainer.models.model_context import ModelContext


def test_derived_scalar_store() -> None:
    activations_by_layer_index: dict[LayerIndex, torch.Tensor] = {
        5: torch.tensor([float(i) for i in range(1, 49, 2)]).view(2, 3, 4),
        6: torch.tensor([float(i) for i in range(2, 50, 2)]).view(2, 3, 4),
    }
    dst = DerivedScalarType.ATTN_QK_PROBS

    ds_store = DerivedScalarStore.from_list(
        [
            ActivationsAndMetadata(
                activations_by_layer_index=activations_by_layer_index,
                dst=dst,
                pass_type=PassType.FORWARD,
            ),
        ]
    )

    layer_5_total = 576
    assert sum(list(range(1, 49, 2))) == layer_5_total
    layer_6_total = 600
    assert sum(list(range(2, 50, 2))) == layer_6_total
    grand_total = layer_5_total + layer_6_total

    assert ds_store.sum() == float(grand_total)

    # Test __getitem__ method
    ds_index = DerivedScalarIndex(
        dst=dst, layer_index=5, tensor_indices=(0, None, None), pass_type=PassType.FORWARD
    )
    result_tensor = ds_store[ds_index]
    expected_first_layer_result: torch.Tensor = activations_by_layer_index[5][0]
    assert isinstance(result_tensor, torch.Tensor)
    assert torch.allclose(
        result_tensor, expected_first_layer_result
    ), f"DerivedScalarStore __getitem__ method failed: {result_tensor} != {expected_first_layer_result}"

    # Test topk method
    top_values, top_indices = ds_store.topk(2)
    assert torch.allclose(
        top_values, torch.tensor([48.0, 47.0])
    ), f"DerivedScalarStore topk method failed: {top_values} != {torch.tensor([48, 46])}"
    assert len(top_indices) == 2
    assert all(
        [
            top_indices[0]
            == DerivedScalarIndex(
                dst=dst, layer_index=6, tensor_indices=(1, 2, 3), pass_type=PassType.FORWARD
            ),
            top_indices[1]
            == DerivedScalarIndex(
                dst=dst, layer_index=5, tensor_indices=(1, 2, 3), pass_type=PassType.FORWARD
            ),
        ]
    )

    # Test apply_transform_fn method
    transform_fn = lambda x: x * 2
    transformed_ds_store = ds_store.apply_transform_fn_to_activations(transform_fn)
    result_tensor = transformed_ds_store[ds_index]
    assert isinstance(result_tensor, torch.Tensor)
    assert torch.allclose(
        result_tensor, activations_by_layer_index[5][0] * 2
    ), "DerivedScalarStore apply_transform_fn method failed"


def test_multi_layer_ds_store() -> None:
    full_activations_by_layer_index: dict[LayerIndex, torch.Tensor] = {
        5: torch.tensor([float(i) for i in range(1, 49, 2)]).view(2, 3, 4),
        6: torch.tensor([float(i) for i in range(2, 50, 2)]).view(2, 3, 4),
    }

    top_values: dict[tuple[int, ...], torch.Tensor] = {}
    top_indices: dict[tuple[int, ...], list[DerivedScalarIndex]] = {}

    for keys_to_include in [(5,), (5, 6), (6,)]:
        activations_by_layer_index: dict[LayerIndex, torch.Tensor] = {
            k: full_activations_by_layer_index[k] for k in keys_to_include
        }
        # define DerivedScalarStore
        ds_store = DerivedScalarStore.from_list(
            [
                ActivationsAndMetadata(
                    activations_by_layer_index=activations_by_layer_index,
                    dst=DerivedScalarType.ATTN_QK_PROBS,
                    pass_type=PassType.FORWARD,
                ),
            ]
        )

        top_values[keys_to_include], top_indices[keys_to_include] = ds_store.topk(2)
        max_value, max_index = ds_store.max()
        assert max_value == top_values[keys_to_include][0]
        assert max_index == top_indices[keys_to_include][0]

    assert torch.allclose(top_values[(5, 6)], torch.tensor([48.0, 47.0]))
    assert torch.allclose(top_values[(5,)], torch.tensor([47.0, 45.0]))
    assert torch.allclose(top_values[(6,)], torch.tensor([48.0, 46.0]))
    assert top_indices[(5, 6)][0] == top_indices[(6,)][0]
    assert top_indices[(5, 6)][1] == top_indices[(5,)][0]


def test_create_ds_store() -> None:
    model_context = ModelContext.from_model_type("gpt2-small", device=torch.device("cpu"))
    num_tokens = 5
    input_shape_by_dim = {
        Dimension.SEQUENCE_TOKENS: num_tokens,
        Dimension.ATTENDED_TO_SEQUENCE_TOKENS: num_tokens,
    }

    dsts = [
        # DerivedScalarType.ATTN_WRITE_NORM,
        DerivedScalarType.MLP_WRITE_NORM,
        # DerivedScalarType.MLP_POST_ACT,
    ]

    def get_shape_for_dim(dim: Dimension) -> int:
        if dim.is_sequence_token_dimension:
            return input_shape_by_dim[dim]
        else:
            return model_context.get_dim_size(dim)

    all_layer_indices = [5, 6]

    all_activations_by_location_type_and_pass_type: (
        dict[ActivationLocationTypeAndPassType, dict[LayerIndex, torch.Tensor]] | None
    ) = None

    def get_subset_of_layer_indices(
        all_activations_by_location_type_and_pass_type: dict[
            ActivationLocationTypeAndPassType, dict[LayerIndex, torch.Tensor]
        ],
        layer_indices: list[LayerIndex],
    ) -> dict[ActivationLocationTypeAndPassType, dict[LayerIndex, torch.Tensor]]:
        return {
            activation_location_type_and_pass_type: {
                layer_index: activations_by_layer_index[layer_index]
                for layer_index in layer_indices
            }
            for activation_location_type_and_pass_type, activations_by_layer_index in all_activations_by_location_type_and_pass_type.items()
        }

    ds_store_by_layer_indices: dict[tuple[int, ...], DerivedScalarStore] = {}

    current_layer_indices_list = [(5, 6), (5,), (6,)]

    for current_layer_indices in current_layer_indices_list:
        dst_config = DstConfig(
            model_context=model_context,
            layer_indices=list(current_layer_indices),
        )

        scalar_derivers = [
            make_scalar_deriver(
                dst=dst,
                dst_config=dst_config,
            )
            for dst in dsts
        ]

        if all_activations_by_location_type_and_pass_type is None:
            # define the raw activations

            all_activations_by_location_type_and_pass_type = {}

            activation_location_type_and_pass_types: list[ActivationLocationTypeAndPassType] = []
            for scalar_deriver in scalar_derivers:
                activation_location_type_and_pass_types.extend(
                    scalar_deriver.get_sub_activation_location_type_and_pass_types()
                )

            activation_location_type_and_pass_types = list(
                set(activation_location_type_and_pass_types)
            )

            for activation_location_type_and_pass_type in activation_location_type_and_pass_types:
                shape_spec = (
                    activation_location_type_and_pass_type.activation_location_type.shape_spec_per_token_sequence
                )

                activation_shape = tuple(get_shape_for_dim(dim) for dim in shape_spec)

                activations_by_layer_index: dict[LayerIndex, torch.Tensor] = {
                    layer_index: torch.randn(activation_shape) for layer_index in all_layer_indices
                }

                all_activations_by_location_type_and_pass_type.update(
                    {activation_location_type_and_pass_type: activations_by_layer_index}
                )

        activations_by_location_type_and_pass_type = get_subset_of_layer_indices(
            all_activations_by_location_type_and_pass_type=all_activations_by_location_type_and_pass_type,
            layer_indices=list(current_layer_indices),
        )

        raw_activation_store = RawActivationStore.from_nested_dict_of_activations(
            activations_by_location_type_and_pass_type
        )

        ds_store_by_layer_indices[current_layer_indices] = DerivedScalarStore.derive_from_raw(
            raw_activation_store=raw_activation_store, scalar_derivers=scalar_derivers
        )

    assert (
        ds_store_by_layer_indices[(5, 6)].sum()
        == ds_store_by_layer_indices[(5,)].sum() + ds_store_by_layer_indices[(6,)].sum()
    )
