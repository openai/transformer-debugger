from neuron_explainer.activations.attention_utils import (
    _inverse_triangular_number,
    convert_flattened_index_to_unflattened_index,
    get_attended_to_sequence_length_per_sequence_token,
    get_max_num_attended_to_sequence_tokens,
)


def _simulate_num_activations(
    num_sequence_tokens: int, max_num_attended_to_sequence_tokens: int
) -> int:
    num_activations_per_token = list(range(1, max_num_attended_to_sequence_tokens + 1)) + [
        max_num_attended_to_sequence_tokens
        for _ in range(num_sequence_tokens - max_num_attended_to_sequence_tokens)
    ]
    num_activations = sum(num_activations_per_token)
    return num_activations


def test_inverse_triangular_number() -> None:
    for m in range(5):
        n = m * (m + 1) // 2
        assert _inverse_triangular_number(n) == m


def test_get_max_num_attended_to_sequence_tokens() -> None:
    num_sequence_tokens = 100
    for max_num_attended_to_sequence_tokens in [50, 100]:
        num_activations = _simulate_num_activations(
            num_sequence_tokens, max_num_attended_to_sequence_tokens
        )
        assert (
            get_max_num_attended_to_sequence_tokens(num_sequence_tokens, num_activations)
            == max_num_attended_to_sequence_tokens
        )

        attended_to_sequence_lengths = get_attended_to_sequence_length_per_sequence_token(
            num_sequence_tokens, max_num_attended_to_sequence_tokens
        )
        assert sum(attended_to_sequence_lengths) == num_activations, (
            sum(attended_to_sequence_lengths),
            num_activations,
        )


def test_convert_flattened_index_to_unflattened_index() -> None:
    possible_max_num_attended_to_sequence_tokens = 9
    num_sequence_tokens = 17
    assert possible_max_num_attended_to_sequence_tokens < num_sequence_tokens
    for max_num_attended_to_sequence_tokens in [
        possible_max_num_attended_to_sequence_tokens,
        num_sequence_tokens,
    ]:
        attended_to_sequence_lengths = get_attended_to_sequence_length_per_sequence_token(
            num_sequence_tokens, max_num_attended_to_sequence_tokens
        )
        num_activations = sum(attended_to_sequence_lengths)

        flat_indices = list(range(num_activations))
        flat_indices_split_by_sequence_token = []
        for attended_to_sequence_length in attended_to_sequence_lengths:
            flat_indices_split_by_sequence_token.append(flat_indices[:attended_to_sequence_length])
            flat_indices = flat_indices[attended_to_sequence_length:]

        for flat_index in list(range(num_activations)):
            if max_num_attended_to_sequence_tokens == num_sequence_tokens:
                unflattened_i, unflattened_j = convert_flattened_index_to_unflattened_index(
                    flat_index
                )
            else:
                unflattened_i, unflattened_j = convert_flattened_index_to_unflattened_index(
                    flat_index,
                    num_sequence_tokens=num_sequence_tokens,
                    num_activations=num_activations,
                )
            assert unflattened_i < num_sequence_tokens
            assert unflattened_j < len(flat_indices_split_by_sequence_token[unflattened_i])
            assert (
                flat_indices_split_by_sequence_token[unflattened_i][unflattened_j] == flat_index
            ), (
                flat_indices_split_by_sequence_token[unflattened_i][unflattened_j],
                flat_index,
            )
