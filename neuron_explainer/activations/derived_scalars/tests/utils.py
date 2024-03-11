from neuron_explainer.activations.derived_scalars.derived_scalar_types import DerivedScalarType
from neuron_explainer.models.model_component_registry import Dimension
from neuron_explainer.models.model_context import ModelContext, get_default_device

get_testing_device = get_default_device  # keep for compatibility


def get_autoencoder_test_path(
    dst: DerivedScalarType,
) -> str:
    """Return the path to a test autoencoder."""

    name = f"{dst.value}.pt"
    return f"az://openaipublic/neuron-explainer/test-data/autoencoder_test_state_dicts/{name}"


def get_activation_shape(
    dst: DerivedScalarType,
    model_context: ModelContext,
    n_tokens: int = 10,
    n_latents: int | None = None,
) -> tuple[int, ...]:
    """Return the shape of activations"""
    activation_shape = []
    assert dst.shape_spec_per_token_sequence[0].is_sequence_token_dimension
    if dst in [
        DerivedScalarType.ATTN_WRITE_NORM,
        DerivedScalarType.FLATTENED_ATTN_POST_SOFTMAX,
        DerivedScalarType.ATTN_ACT_TIMES_GRAD,
        DerivedScalarType.ATTN_WRITE_TO_FINAL_RESIDUAL_GRAD,
    ]:
        # first dimension is token pairs
        activation_shape.append(n_tokens * (n_tokens + 1) // 2)
    else:
        activation_shape.append(n_tokens)
    for dimension in dst.shape_spec_per_token_sequence[1:]:
        if dimension == Dimension.SINGLETON:
            activation_shape.append(1)
        elif dimension.is_model_intrinsic:
            activation_shape.append(model_context.get_dim_size(dimension))
        elif dimension.is_sequence_token_dimension:
            activation_shape.append(n_tokens)
        elif dimension.is_parameterized_dimension:
            assert n_latents is not None
            activation_shape.append(n_latents)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    print(f"{dst}: {activation_shape}")
    return tuple(activation_shape)
