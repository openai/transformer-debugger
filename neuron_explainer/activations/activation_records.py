"""Utilities for formatting activation records into prompts."""

import math
from typing import Any, Sequence

from neuron_explainer.activations.activations import ActivationRecord

UNKNOWN_ACTIVATION_STRING = "unknown"


def relu(x: float) -> float:
    return max(0.0, x)


def calculate_max_activation(activation_records: Sequence[ActivationRecord]) -> float:
    """Return the maximum activation value of the neuron across all the activation records."""
    flattened = [
        # Relu is used to assume any values less than 0 are indicating the neuron is in the resting
        # state.
        max(relu(x) for x in activation_record.activations)
        for activation_record in activation_records
    ]
    return max(flattened)


def normalize_activations(activation_record: list[float], max_activation: float) -> list[int]:
    """Convert raw neuron activations to integers on the range [0, 10]."""
    if max_activation <= 0:
        # raise ValueError(f"max_activation must be positive, got {max_activation}")
        return [
            0 for x in activation_record
        ]  # commented the above line to allow GPT2 datasets to render (Dan)
    # Relu is used to assume any values less than 0 are indicating the neuron is in the resting
    # state.
    return [min(10, math.floor(10 * relu(x) / max_activation)) for x in activation_record]


def normalize_activations_symmetric(
    activation_record: list[float], max_activation: float
) -> list[int]:
    """Convert raw neuron activations to integers on the range [-10, 10]."""
    max_abs_activation = (
        max_activation  # clients expect kwarg "max_activation", so leaving this for now
    )
    if max_abs_activation == 0.0:
        # raise ValueError(f"max_activation must be positive, got {max_activation}")
        return [0 for x in activation_record]
    # Unlike normalize_activations, this function doesn't apply relu, since we want to show negative
    # activations as well.
    return [max(min(10, math.trunc(10 * x / max_abs_activation)), -10) for x in activation_record]


def truncate_negative_activations(activation_record: ActivationRecord) -> ActivationRecord:
    """Truncate activations to 0 if they are negative."""
    return ActivationRecord(
        tokens=activation_record.tokens,
        activations=[max(0, x) for x in activation_record.activations],
    )


def truncate_negative_activations_list(
    activation_records: Sequence[ActivationRecord],
) -> list[ActivationRecord]:
    """Truncate activations to 0 if they are negative."""
    return [truncate_negative_activations(x) for x in activation_records]


def _format_activation_record(
    activation_record: ActivationRecord,
    max_activation: float,
    omit_zeros: bool,
    hide_activations: bool = False,
    start_index: int = 0,
) -> str:
    """Format neuron activations into a string, suitable for use in prompts."""
    tokens = activation_record.tokens
    normalized_activations = normalize_activations(activation_record.activations, max_activation)
    if omit_zeros:
        assert (not hide_activations) and start_index == 0, "Can't hide activations and omit zeros"
        tokens = [
            token for token, activation in zip(tokens, normalized_activations) if activation > 0
        ]
        normalized_activations = [x for x in normalized_activations if x > 0]

    entries = []
    assert len(tokens) == len(normalized_activations)
    for index, token, activation in zip(range(len(tokens)), tokens, normalized_activations):
        activation_string = str(int(activation))
        if hide_activations or index < start_index:
            activation_string = UNKNOWN_ACTIVATION_STRING
        entries.append(f"{token}\t{activation_string}")
    return "\n".join(entries)


def format_activation_records(
    activation_records: Sequence[ActivationRecord],
    max_activation: float,
    *,
    omit_zeros: bool = False,
    start_indices: list[int] | None = None,
    hide_activations: bool = False,
) -> str:
    """Format a list of activation records into a string."""
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(
            [
                _format_activation_record(
                    activation_record,
                    max_activation,
                    omit_zeros=omit_zeros,
                    hide_activations=hide_activations,
                    start_index=0 if start_indices is None else start_indices[i],
                )
                for i, activation_record in enumerate(activation_records)
            ]
        )
        + "\n<end>\n"
    )


def _format_tokens_for_simulation(tokens: Sequence[str]) -> str:
    """
    Format tokens into a string with each token marked as having an "unknown" activation, suitable
    for use in prompts.
    """
    entries = []
    for token in tokens:
        entries.append(f"{token}\t{UNKNOWN_ACTIVATION_STRING}")
    return "\n".join(entries)


def format_sequences_for_simulation(
    all_tokens: Sequence[Sequence[str]],
) -> str:
    """
    Format a list of lists of tokens into a string with each token marked as having an "unknown"
    activation, suitable for use in prompts.
    """
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(
            [_format_tokens_for_simulation(tokens) for tokens in all_tokens]
        )
        + "\n<end>\n"
    )


def non_zero_activation_proportion(
    activation_records: Sequence[ActivationRecord], max_activation: float
) -> float:
    """Return the proportion of activation values that aren't zero."""
    total_activations_count = sum(
        [len(activation_record.activations) for activation_record in activation_records]
    )
    normalized_activations = [
        normalize_activations(activation_record.activations, max_activation)
        for activation_record in activation_records
    ]
    non_zero_activations_count = sum(
        [len([x for x in activations if x != 0]) for activations in normalized_activations]
    )
    return non_zero_activations_count / total_activations_count


def get_attribute_or_key(activation_record: Any, attribute_name: str) -> Any:
    if isinstance(activation_record, dict):
        assert attribute_name in activation_record, f"{attribute_name} not in activation_record"
        return activation_record[attribute_name]
    else:
        assert hasattr(activation_record, attribute_name)
        return getattr(activation_record, attribute_name)
