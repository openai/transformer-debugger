"""
This file contains utilities and a class for converting token-space vectors to and from a pydantic
base class summarizing them in terms of the extremal entries in the vector and their associated
tokens. The class can be used in InteractiveModel responses.
"""

import math

import torch
from pydantic import validator

from neuron_explainer.activations.derived_scalars.least_common_tokens import (
    LEAST_COMMON_GPT2_TOKEN_STRS,
)
from neuron_explainer.models.model_context import ModelContext
from neuron_explainer.pydantic import CamelCaseBaseModel, immutable


@immutable
class TokenAndRawScalar(CamelCaseBaseModel):
    token: str
    scalar: float

    @validator("scalar")
    def check_scalar(cls, scalar: float) -> float:
        assert math.isfinite(scalar), "Scalar value must be a finite number"
        return scalar


@immutable
class TokenAndScalar(TokenAndRawScalar):
    normalized_scalar: float

    @validator("normalized_scalar")
    def check_normalized_scalar(cls, normalized_scalar: float) -> float:
        assert math.isfinite(normalized_scalar), "Normalized scalar value must be a finite number"
        return normalized_scalar


@immutable
class TopTokens(CamelCaseBaseModel):
    """
    Contains two lists of tokens and associated scalars: one for the highest-scoring tokens and one
    for the lowest-scoring tokens, according to some way of scoring tokens. For example, this could
    be used to represent the top upvoted and downvoted "logit lens" tokens. An instance of this
    class is scoped to a single node. The set of tokens eligible for scoring is typically just the
    model's entire vocabulary. Each list is sorted from largest to smallest absolute value for the
    associated scalar.
    """

    top: list[TokenAndScalar]
    bottom: list[TokenAndScalar]


def package_top_t_tokens(
    model_context: ModelContext,
    top_t_upvoted_token_ints_tensor: torch.Tensor,
    top_t_upvoted_token_weights_tensor: torch.Tensor,
    norm_top_t_upvoted_token_weights_tensor: torch.Tensor,
) -> list[list[TokenAndScalar]]:
    """
    Convert tensors of top t upvoted token ints, weights, and normalized weights into a list of
    lists of TokenAndScalar, one list per node.
    """
    n_nodes, n_tokens = top_t_upvoted_token_ints_tensor.shape
    top_t_upvoted_token_strings = [
        model_context.decode_token_list(top_t_upvoted_token_ints_tensor[i].tolist())
        for i in range(top_t_upvoted_token_ints_tensor.shape[0])
    ]
    top_t_upvoted_token_weights = top_t_upvoted_token_weights_tensor.tolist()
    norm_top_t_upvoted_token_weights = norm_top_t_upvoted_token_weights_tensor.tolist()
    token_and_weight_data_for_all_nodes = []
    # for each row of the tensor, zip the results into a list of TokenAndRawScalar for the relevant node
    for node_index in range(n_nodes):
        token_and_weight_data_for_this_node = []
        # zip the results into a list of TokenAndRawScalar for this node
        for token_index in range(n_tokens):
            token_and_weight_data_for_this_node.append(
                TokenAndScalar(
                    token=top_t_upvoted_token_strings[node_index][token_index],
                    scalar=top_t_upvoted_token_weights[node_index][token_index],
                    normalized_scalar=norm_top_t_upvoted_token_weights[node_index][token_index],
                )
            )
        token_and_weight_data_for_all_nodes.append(token_and_weight_data_for_this_node)
    return token_and_weight_data_for_all_nodes


def get_top_t_tokens_maybe_excluding_least_common(
    token_writes_tensor: torch.Tensor,
    top_t_tokens: int,
    largest: bool,
    least_common_tokens_as_ints: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the top t tokens and their weights, optionally excluding the least common token ints,
    passed as an argument. Sorted by largest or smallest absolute value, as specified by the
    'largest' argument.
    """
    if least_common_tokens_as_ints is not None:
        token_writes_tensor[:, least_common_tokens_as_ints] = (
            float("-inf") if largest else float("inf")
        )
    (
        top_t_upvoted_token_weights_tensor,
        top_t_upvoted_token_ints_tensor,
    ) = token_writes_tensor.topk(k=top_t_tokens, largest=largest)
    assert torch.isfinite(
        top_t_upvoted_token_weights_tensor
    ).all(), "Top token weights should only contain finite values"
    return top_t_upvoted_token_weights_tensor, top_t_upvoted_token_ints_tensor


def get_most_upvoted_and_downvoted_tokens_for_nodes(
    model_context: ModelContext,
    token_writes_tensor: torch.Tensor,
    top_t_tokens: int,
    flip_upvoted_and_downvoted: bool = False,
) -> list[TopTokens]:
    """
    Convert a 2D token_writes_tensor to the most positive (upvoted) and most negative (downvoted) vocab tokens per row,
    with weights corresponding to how upvoted or downvoted each token is. Return a list (indexed by row index) of
    TopTokens, each of which contains a list of TokenAndScalar for the most upvoted tokens and
    a list of TokenAndScalar for the most downvoted tokens.

    Note that the scalars in TokenAndScalar are referred to as 'weights', despite being held in an object called
    TokenAndScalar. The weights returned here include normalized versions (normalized to max(abs(weight))).
    """

    if model_context.get_encoding().name == "gpt2":
        # for GPT-2, we exclude tokens string-matching to the least common tokens
        # from the top_t tokens displayed
        least_common_tokens_as_ints = model_context.encode_token_str_list(
            LEAST_COMMON_GPT2_TOKEN_STRS
        )
    else:
        least_common_tokens_as_ints = None
    (
        top_t_upvoted_token_weights_tensor,
        top_t_upvoted_token_ints_tensor,
    ) = get_top_t_tokens_maybe_excluding_least_common(
        token_writes_tensor,
        top_t_tokens,
        largest=True,
        least_common_tokens_as_ints=least_common_tokens_as_ints,
    )
    (
        top_t_downvoted_token_weights_tensor,
        top_t_downvoted_token_ints_tensor,
    ) = get_top_t_tokens_maybe_excluding_least_common(
        token_writes_tensor,
        top_t_tokens,
        largest=False,
        least_common_tokens_as_ints=least_common_tokens_as_ints,
    )
    max_abs_token_writes_tensor = torch.max(
        top_t_upvoted_token_weights_tensor[:, 0:1].abs(),
        top_t_downvoted_token_weights_tensor[:, 0:1].abs(),
    )

    def safe_divide(
        numerator_tensor: torch.Tensor, denominator_tensor: torch.Tensor
    ) -> torch.Tensor:
        assert torch.isfinite(
            numerator_tensor
        ).all(), "Numerator tensor should only contain finite values"
        assert torch.isfinite(
            denominator_tensor
        ).all(), "Denominator tensor should only contain finite values"
        return torch.where(
            denominator_tensor == 0,
            torch.zeros_like(numerator_tensor),
            numerator_tensor / denominator_tensor,
        )

    norm_top_t_upvoted_token_weights_tensor = safe_divide(
        top_t_upvoted_token_weights_tensor, max_abs_token_writes_tensor
    )
    norm_top_t_downvoted_token_weights_tensor = safe_divide(
        top_t_downvoted_token_weights_tensor, max_abs_token_writes_tensor
    )

    normalized_most_upvoted_tokens_for_all_nodes = package_top_t_tokens(
        model_context,
        top_t_upvoted_token_ints_tensor,
        top_t_upvoted_token_weights_tensor,
        norm_top_t_upvoted_token_weights_tensor,
    )
    normalized_most_downvoted_tokens_for_all_nodes = package_top_t_tokens(
        model_context,
        top_t_downvoted_token_ints_tensor,
        top_t_downvoted_token_weights_tensor,
        norm_top_t_downvoted_token_weights_tensor,
    )
    # zip the results into a list of TopTokens
    top_tokens_list = []
    for node_index in range(len(normalized_most_upvoted_tokens_for_all_nodes)):
        if flip_upvoted_and_downvoted:
            normalized_most_upvoted_tokens_for_node = (
                normalized_most_downvoted_tokens_for_all_nodes[node_index]
            )
            normalized_most_downvoted_tokens_for_node = (
                normalized_most_upvoted_tokens_for_all_nodes[node_index]
            )
        else:
            normalized_most_upvoted_tokens_for_node = normalized_most_upvoted_tokens_for_all_nodes[
                node_index
            ]
            normalized_most_downvoted_tokens_for_node = (
                normalized_most_downvoted_tokens_for_all_nodes[node_index]
            )
        top_tokens_list.append(
            TopTokens(
                top=normalized_most_upvoted_tokens_for_node,
                bottom=normalized_most_downvoted_tokens_for_node,
            )
        )

    assert len(top_tokens_list) == token_writes_tensor.shape[0]
    return top_tokens_list
