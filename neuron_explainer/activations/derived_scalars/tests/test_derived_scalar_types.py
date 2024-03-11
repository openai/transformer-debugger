import blobfile as bf
import pytest
import torch

from neuron_explainer.activations.derived_scalars.attention import (
    flatten_lower_triangle,
    make_reshape_fn,
)
from neuron_explainer.activations.derived_scalars.autoencoder import (
    make_autoencoder_latent_scalar_deriver_factory,
    make_autoencoder_write_norm_scalar_deriver_factory,
)
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    ActivationsAndMetadata,
    DerivedScalarType,
    DstConfig,
    ScalarSource,
)
from neuron_explainer.activations.derived_scalars.tests.utils import (
    get_activation_shape,
    get_autoencoder_test_path,
)
from neuron_explainer.models import Autoencoder
from neuron_explainer.models.autoencoder_context import AutoencoderConfig, AutoencoderContext
from neuron_explainer.models.model_component_registry import NodeType, PassType
from neuron_explainer.models.model_context import ModelContext, get_default_device


def test_flatten_lower_triangle() -> None:
    """This tests code which turns the lower triangular entries tensor (that is, entries with
    i <= j for i and j indexing the last two dimensions) into a tensor flattened along the last
    two dimensions."""
    # Test the function with a (2, 2, 4, 4) tensor
    t = torch.arange(1, 65).reshape(2, 2, 4, 4)
    flattened_t = flatten_lower_triangle(t)
    expected_t = torch.tensor(
        [
            [[1, 5, 6, 9, 10, 11, 13, 14, 15, 16], [17, 21, 22, 25, 26, 27, 29, 30, 31, 32]],
            [[33, 37, 38, 41, 42, 43, 45, 46, 47, 48], [49, 53, 54, 57, 58, 59, 61, 62, 63, 64]],
        ],
        dtype=torch.int64,
    )
    assert flattened_t.shape == (2, 2, 10), f"Got shape {flattened_t.shape}, expected (2, 2, 10)"
    assert torch.allclose(
        flattened_t, expected_t
    ), "Flattened tensor values do not match expected values"

    # Test the function with a (3, 2, 2, 4, 4) tensor
    t = torch.arange(1, 33).reshape(2, 4, 4)
    flattened_t = flatten_lower_triangle(t)
    expected_t = torch.tensor(
        [[1, 5, 6, 9, 10, 11, 13, 14, 15, 16], [17, 21, 22, 25, 26, 27, 29, 30, 31, 32]],
        dtype=torch.int64,
    )
    assert flattened_t.shape == (2, 10), f"Got shape {flattened_t.shape}, expected (2, 2, 10)"
    assert torch.allclose(
        flattened_t, expected_t
    ), "Flattened tensor values do not match expected values"


def test_dst_values_equal_to_names() -> None:
    message = "DerivedScalarType values should be equal to their names (lowercased)."
    for dst in DerivedScalarType:
        assert dst.value == dst.name.lower(), message


def make_fake_activations_and_metadata_tuple(
    sub_scalar_sources: tuple[ScalarSource, ...],
    model_context: ModelContext,
    n_tokens: int,
    n_latents: int | None = None,
) -> tuple[ActivationsAndMetadata, ...]:
    activations_and_metadata_list = []
    desired_layer_indices = None
    for sub_scalar_source in sub_scalar_sources:
        dst_and_pass_type = sub_scalar_source.dst_and_pass_type
        layer_indexer = sub_scalar_source.layer_indexer
        this_dst = dst_and_pass_type.dst
        activation_shape = get_activation_shape(this_dst, model_context, n_tokens, n_latents)
        activations_and_metadata = ActivationsAndMetadata(
            activations_by_layer_index={
                layer_index: torch.randn(activation_shape, device=model_context.device)
                for layer_index in range(model_context.n_layers)
            },
            pass_type=PassType.FORWARD,
            dst=this_dst,
        ).apply_layer_indexer(layer_indexer, desired_layer_indices)
        activations_and_metadata_list += [activations_and_metadata]
        if len(activations_and_metadata_list) == 1:
            # match the layer_indices of the first activations_and_metadata object
            desired_layer_indices = list(activations_and_metadata_list[0].layer_indices)
    return tuple(activations_and_metadata_list)


def _get_autoencoder_test_path_maybe_saving_new_autoencoder(
    dst: DerivedScalarType,
    model_context: ModelContext,
) -> str:
    """Return the path to a test autoencoder. If the autoencoder does not exist, create it."""

    autoencoder_path = get_autoencoder_test_path(dst)

    if bf.exists(autoencoder_path):
        return autoencoder_path

    # create autoencoder
    activation_shape = get_activation_shape(dst, model_context)
    input_tensor = torch.zeros(activation_shape)
    reshape_fn = make_reshape_fn(dst)
    reshaped_input_tensor = reshape_fn(input_tensor)
    print(f"autoencoder_input_tensor.shape: {reshaped_input_tensor.shape}")
    n_inputs = reshaped_input_tensor.shape[1]
    n_latents = 6
    autoencoder = Autoencoder(n_latents, n_inputs)
    print({autoencoder.pre_bias.shape})

    # change bias to have non-zero latent activations, to have non-zero gradients in test
    autoencoder.latent_bias.data[:] = 100.0

    # save autoencoder
    with bf.BlobFile(autoencoder_path, "wb") as f:
        print(f"Saving autoencoder to {autoencoder_path}")
        torch.save(autoencoder.state_dict(), f)

    return autoencoder_path


@pytest.mark.parametrize(
    "dst",
    [
        DerivedScalarType.MLP_POST_ACT,
        DerivedScalarType.RESID_DELTA_MLP_FROM_MLP_POST_ACT,
        DerivedScalarType.ATTN_WEIGHTED_SUM_OF_VALUES,
        DerivedScalarType.RESID_DELTA_ATTN,
        DerivedScalarType.ATTN_WEIGHTED_VALUE,
        DerivedScalarType.ATTN_WRITE,
    ],
)
def test_autoencoder_scalar_deriver(
    dst: DerivedScalarType,
) -> None:
    model_name = "gpt2-small"
    device = get_default_device()
    model_context = ModelContext.from_model_type(model_name, device=device)
    layer_indices = list(range(model_context.n_layers))
    n_tokens = 10

    # create autoencoder config
    autoencoder_path = _get_autoencoder_test_path_maybe_saving_new_autoencoder(dst, model_context)
    autoencoder_config = AutoencoderConfig(
        dst=dst,
        autoencoder_path_by_layer_index={
            layer_index: autoencoder_path for layer_index in layer_indices
        },
    )
    autoencoder_context = AutoencoderContext(autoencoder_config=autoencoder_config, device=device)
    autoencoder_context.warmup()
    n_latents = autoencoder_context.num_autoencoder_directions

    # Test
    for make_scalar_deriver_factory, generic_dst in zip(
        [
            make_autoencoder_latent_scalar_deriver_factory,
            make_autoencoder_write_norm_scalar_deriver_factory,
        ],
        [
            DerivedScalarType.AUTOENCODER_LATENT,
            DerivedScalarType.AUTOENCODER_WRITE_NORM,
        ],
    ):
        if generic_dst == DerivedScalarType.AUTOENCODER_WRITE_NORM and (
            dst.node_type != NodeType.RESIDUAL_STREAM_CHANNEL
        ):
            continue

        # create scalar deriver
        dst_config = DstConfig(
            layer_indices=layer_indices,
            model_name=model_name,
            autoencoder_context=autoencoder_context,
        )
        scalar_deriver = make_scalar_deriver_factory(autoencoder_context.autoencoder_node_type)(
            dst_config
        )

        # create fake dataset of activations
        activations_and_metadata_tuple = make_fake_activations_and_metadata_tuple(
            scalar_deriver.get_sub_scalar_sources(), model_context, n_tokens, n_latents
        )

        # calculate derived scalar
        new_activations_and_metadata = (
            scalar_deriver.activations_and_metadata_calculate_derived_scalar_fn(
                activations_and_metadata_tuple, PassType.FORWARD
            )
        )
        assert list(new_activations_and_metadata.activations_by_layer_index.keys()) == layer_indices
        assert (
            new_activations_and_metadata.activations_by_layer_index[layer_indices[0]].shape[1]
            == n_latents
        )
        # saved memory by not storing gradients -> is_leaf
        assert new_activations_and_metadata.activations_by_layer_index[layer_indices[0]].is_leaf

        node_specific_dst = generic_dst.update_from_autoencoder_node_type(
            autoencoder_context.autoencoder_node_type
        )
        assert new_activations_and_metadata.dst == node_specific_dst
