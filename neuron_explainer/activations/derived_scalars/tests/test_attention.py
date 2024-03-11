import pytest
import torch

from neuron_explainer.activations.derived_scalars.attention import (
    flatten_lower_triangle,
    unflatten_lower_triangle,
    unflatten_lower_triangle_and_sum_columns,
)


@pytest.mark.parametrize("extra_dim", [[], [2], [2, 3]])
@pytest.mark.parametrize("N", [63, 64, 65])
def test_unflatten_lower_triangle(extra_dim: list[int], N: int) -> None:
    """Test that unflatten_lower_triangle is the inverse of flatten_lower_triangle."""
    # Create a random tensor of shape ... x M x N
    M = 64
    original_tensor = torch.rand(extra_dim + [M, N])

    # Set all elements above the lower triangular to 0
    lower_triangular_mask = torch.tril(torch.ones(M, N)).bool()
    original_tensor[..., ~lower_triangular_mask] = 0

    # Apply flatten_lower_triangle to the original tensor
    flattened = flatten_lower_triangle(original_tensor)
    assert flattened.shape == tuple(extra_dim + [lower_triangular_mask.sum()])

    # Apply unflatten_lower_triangle to the flattened tensor
    reconstructed_tensor = unflatten_lower_triangle(flattened, M, N)
    assert torch.allclose(original_tensor, reconstructed_tensor)


@pytest.mark.parametrize("extra_dim", [[], [2], [2, 3]])
@pytest.mark.parametrize("N", [63, 64, 65])
def test_unflatten_lower_triangle_and_sum_columns(extra_dim: list[int], N: int) -> None:
    """Test unflatten_lower_triangle_and_sum_columns(...) is equal to unflatten_lower_triangle(...).sum(-1)."""
    # Create a random flattened tensor
    M = 64
    num_elements = int(torch.tril(torch.ones(M, N)).bool().sum().item())
    flattened = torch.rand(extra_dim + [num_elements])

    # apply unflatten_lower_triangle_and_sum_columns
    result = unflatten_lower_triangle_and_sum_columns(flattened, M, N)

    # apply unflatten_lower_triangle and sum(-1)
    reconstructed = unflatten_lower_triangle(flattened, M, N)
    reference = reconstructed.sum(dim=-1)
    assert torch.allclose(result, reference)
