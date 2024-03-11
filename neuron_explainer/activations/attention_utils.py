"""
Contains math utilities for converting from flattened representations of attention activations
(which are a scalar per token pair) to nested lists. The inner lists are attention activations
related to attention from the same token (to different tokens).

Tested in ./test_attention_utils.py.
"""

import math

import numpy as np


def _inverse_triangular_number(n: int) -> int:
    # the m'th triangular number t_m satisfies t_m = m(m+1)/2
    # this function asserts that n is a triangular number, and returns the unique m such that t_m = n
    # this is used to infer the number of sequence tokens from the number of activations
    assert n >= 0
    m: int = (
        math.floor(math.sqrt(1 + 8 * n)) - 1
    ) // 2  # from quadratic formula applied to m(m+1)/2 = n
    assert m * (m + 1) // 2 == n
    return m


def get_max_num_attended_to_sequence_tokens(num_sequence_tokens: int, num_activations: int) -> int:
    # Attended to sequences are assumed to increase in length up to a maximum length, and then stay at that
    # length for the remainder of the sequence. The maximum attended to sequence length is at most equal to the sequence length,
    # but is permitted to be less
    num_sequence_token_pairs = num_sequence_tokens * (num_sequence_tokens + 1) // 2
    if num_activations == num_sequence_token_pairs:
        # the maximum attended to sequence length is equal to the sequence length
        return num_sequence_tokens
    else:
        # the maximum attended to sequence length is less than the sequence length, and
        assert num_activations < num_sequence_token_pairs
        num_missing_activations = num_sequence_token_pairs - num_activations
        num_missing_sequence_tokens = _inverse_triangular_number(num_missing_activations)
        max_num_attended_to_sequence_tokens = num_sequence_tokens - num_missing_sequence_tokens
        assert max_num_attended_to_sequence_tokens > 0
        return max_num_attended_to_sequence_tokens


def get_attended_to_sequence_length_per_sequence_token(
    num_sequence_tokens: int, max_num_attended_to_sequence_tokens: int
) -> list[int]:
    # given a num_sequence_tokens and a max_num_attended_to_sequence_tokens, return a list of length num_sequence_tokens
    # where the ith element is the length of the attended to sequence for the ith sequence token.
    # The length of the attended to sequence starts at 1, increases up to max_num_attended_to_sequence_tokens, by 1 with each
    # token, and then stays at max_num_attended_to_sequence_tokens for the remainder of the sequence
    assert num_sequence_tokens >= max_num_attended_to_sequence_tokens
    attended_to_sequence_lengths = list(range(1, max_num_attended_to_sequence_tokens + 1))
    if num_sequence_tokens > max_num_attended_to_sequence_tokens:
        attended_to_sequence_lengths.extend(
            [
                max_num_attended_to_sequence_tokens
                for _ in range(num_sequence_tokens - max_num_attended_to_sequence_tokens)
            ]
        )
    return attended_to_sequence_lengths


def get_attended_to_sequence_lengths(num_sequence_tokens: int, num_activations: int) -> list[int]:
    max_num_attended_to_sequence_tokens = get_max_num_attended_to_sequence_tokens(
        num_sequence_tokens, num_activations
    )
    return get_attended_to_sequence_length_per_sequence_token(
        num_sequence_tokens, max_num_attended_to_sequence_tokens
    )


def _convert_flattened_index_to_unflattened_index_assuming_square_matrix(
    flat_index: int,
) -> tuple[int, int]:
    # this con
    n = math.floor((-1 + math.sqrt(1 + 8 * flat_index)) / 2)
    m = flat_index - n * (n + 1) // 2
    return n, m


def convert_flattened_index_to_unflattened_index(
    flattened_index: int,
    num_sequence_tokens: int | None = None,
    num_activations: int | None = None,
) -> tuple[int, int]:
    # given a flattened index, return the unflattened index
    # if the attention matrix is square (most common), then the flattened_index uniquely determines the index within the square matrix
    # if the attention matrix has more rows (sequence tokens) than columns (attended-to sequence tokens), then num_sequence_tokens
    # and num_activations are required to determine the index within the matrix
    # specify both num_sequence_tokens and num_activations, or neither
    assert not (num_sequence_tokens is None) ^ (num_activations is None)

    if (
        num_sequence_tokens is None
        or num_activations == num_sequence_tokens * (num_sequence_tokens + 1) // 2
    ):
        assume_square_matrix = True
    else:
        assume_square_matrix = False

    if assume_square_matrix:
        return _convert_flattened_index_to_unflattened_index_assuming_square_matrix(flattened_index)
    else:
        assert num_sequence_tokens is not None
        assert num_activations is not None
        assert flattened_index < num_activations
        sequence_lengths = get_attended_to_sequence_lengths(num_sequence_tokens, num_activations)
        sequence_lengths_cumsum = np.cumsum([0] + sequence_lengths)
        sequence_index = int(
            np.searchsorted(sequence_lengths_cumsum, flattened_index, side="right") - 1
        )
        assert sequence_lengths_cumsum[sequence_index] <= flattened_index, (
            sequence_lengths_cumsum[sequence_index],
            flattened_index,
        )
        assert sequence_lengths_cumsum[sequence_index + 1] >= flattened_index, (
            sequence_lengths_cumsum[sequence_index + 1],
            flattened_index,
        )
        index_within_sequence = flattened_index - sequence_lengths_cumsum[sequence_index]
        return sequence_index, index_within_sequence
