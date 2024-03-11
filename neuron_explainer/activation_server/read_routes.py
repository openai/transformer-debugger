"""Routes / endpoints related to reading existing data from Azure blob storage."""

from __future__ import annotations

import asyncio
from typing import Callable, Sequence, TypeVar

from fastapi import FastAPI, HTTPException

from neuron_explainer.activation_server.explanation_datasets import get_all_explanation_datasets
from neuron_explainer.activation_server.load_neurons import (
    NodeIdAndDatasets,
    load_neuron_from_datasets,
    resolve_neuron_dataset,
)
from neuron_explainer.activation_server.neuron_datasets import (
    NeuronDatasetMetadata,
    get_all_neuron_dataset_metadata,
)
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.activations.attention_utils import get_attended_to_sequence_lengths
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.tokens import TokenAndRawScalar, TokenAndScalar
from neuron_explainer.explanations.explanations import (
    ScoredExplanation,
    load_neuron_explanations_async,
)
from neuron_explainer.pydantic import CamelCaseBaseModel, immutable

T = TypeVar("T")


@immutable
class AttributedScoredExplanation(CamelCaseBaseModel):
    explanation: str
    score: float | None
    """
    None means that the explanation's score is undefined, maybe because the neuron is never active.
    """
    dataset_name: str


@immutable
class ExistingExplanationsRequest(CamelCaseBaseModel):
    dst: DerivedScalarType
    layer_index: int
    activation_index: int
    # Exactly one of explanation_datasets and neuron_dataset must be specified. If
    # explanation_datasets is specified, only those datasets will be used. If neuron_dataset is
    # specified, all of its explanation datasets will be used.
    explanation_datasets: list[str]
    neuron_dataset: str | None


@immutable
class TokenAndRawAttentionScalars(CamelCaseBaseModel):
    # same as TokenAndAttentionScalars, but without any of the postprocessing
    # applied
    token: str
    scalars: list[float]


@immutable
class TokenAndAttentionScalars(TokenAndRawAttentionScalars):
    # attention_scalars can be any quantity that is a scalar per attention head
    # and per token pair. The most straightforward examples are pre- and post-softmax
    # attention scores; norm of attention head write vector is another option

    # Causally masked attention results in attention scalars for each pair of
    # tokens, which read from an earlier token and write to a later token. The
    # TokenAndAttentionScalars object for the nth token in a sequence contains
    # a list of n floats, summarizing in some way the values written to the nth
    # (current) token by reading from all n-1 earlier tokens, plus the current
    # (nth) token. Because the things read and written are vectors rather than
    # scalars, there are presumably many ways to summarize them as scalars;
    # this object contains just one kind of summary.

    normalized_scalars: list[float]
    total_scalar_in: float
    normalized_total_scalar_in: float
    max_scalar_in: float
    normalized_max_scalar_in: float
    total_scalar_out: float
    normalized_total_scalar_out: float
    max_scalar_out: float
    normalized_max_scalar_out: float


@immutable
class NeuronRecordResponse(CamelCaseBaseModel):
    dataset: str
    max_activation: float
    top_activations: list[list[TokenAndScalar]]
    random_sample: list[list[TokenAndScalar]]


@immutable
class AttentionHeadRecordResponse(CamelCaseBaseModel):
    dataset: str
    max_attention_activation: float
    most_positive_token_sequences: list[list[TokenAndAttentionScalars]]
    random_token_sequences: list[list[TokenAndAttentionScalars]]


def zip_tokens_and_activations(
    activation_records: list[ActivationRecord],
) -> list[list[TokenAndRawScalar]]:
    sequences = []
    for activation_record in activation_records:
        sequence = []
        for token, activation in zip(activation_record.tokens, activation_record.activations):
            sequence.append(
                TokenAndRawScalar(
                    token=token,
                    scalar=activation,
                )
            )
        sequences.append(sequence)
    return sequences


def convert_activation_records_to_token_and_activation_lists(
    all_activation_record_lists: list[list[ActivationRecord]],
) -> list[list[list[TokenAndScalar]]]:
    zipped_tokens_and_raw_activations = [
        zip_tokens_and_activations(activation_record_list)
        for activation_record_list in all_activation_record_lists
    ]
    return normalize_token_scalars(zipped_tokens_and_raw_activations)


def zip_tokens_and_attention_activations(
    activation_records: list[ActivationRecord],
) -> list[list[TokenAndRawAttentionScalars]]:
    """
    This function takes a list of activation records and returns a list of lists of TokenAndRawAttentionScalars, one
    outer list element per sequence, and one inner list element per sequence token.
    There are multiple attention activations per sequence token (one per token in the "attended to sequence" for that sequence token).
    Because the activation record holds the activations in a flattened form, where (sequence token index, attended to sequence token index)
    are combined into a single index, we need to reconstruct the nested list structure, inferring it from the number of flattened
    activations and the number of sequence tokens.
    """

    sequences: list[list[TokenAndRawAttentionScalars]] = []
    for activation_record in activation_records:
        sequence = convert_token_sequence_to_token_and_raw_attention_scalars(
            activation_record.tokens, activation_record.activations
        )
        sequences.append(sequence)
    return sequences


def convert_token_sequence_to_token_and_raw_attention_scalars(
    token_strings: list[str], activations: list[float]
) -> list[TokenAndRawAttentionScalars]:
    num_sequence_tokens = len(token_strings)
    num_activations = len(activations)
    attended_to_sequence_lengths_indexed_by_sequence_token = get_attended_to_sequence_lengths(
        num_sequence_tokens, num_activations
    )
    attention_activations_list_indexed_by_sequence_token = []
    start = 0
    for (
        attended_to_sequence_length_for_sequence_token
    ) in attended_to_sequence_lengths_indexed_by_sequence_token:
        end = start + attended_to_sequence_length_for_sequence_token
        attention_activations_list_indexed_by_sequence_token.append(activations[start:end])
        start = end
    assert len(attention_activations_list_indexed_by_sequence_token) == len(token_strings)

    sequence = []
    for token_string, activations in zip(
        token_strings, attention_activations_list_indexed_by_sequence_token
    ):
        sequence.append(
            TokenAndRawAttentionScalars(
                token=token_string,
                scalars=activations,
            )
        )
    return sequence


def convert_activation_records_to_token_and_attention_activations_lists(
    all_activation_record_lists: list[list[ActivationRecord]],
) -> list[list[list[TokenAndAttentionScalars]]]:
    zipped_tokens_and_raw_attention_activations = [
        zip_tokens_and_attention_activations(activation_record_list)
        for activation_record_list in all_activation_record_lists
    ]
    return normalize_attention_token_scalars(zipped_tokens_and_raw_attention_activations)


def define_read_routes(app: FastAPI) -> None:
    @app.post(
        "/existing_explanations", response_model=list[AttributedScoredExplanation], tags=["read"]
    )
    async def existing_explanations(
        request: ExistingExplanationsRequest,
    ) -> list[AttributedScoredExplanation]:
        def convert_scored_explanation(
            scored_explanation: ScoredExplanation, explanation_dataset: str
        ) -> AttributedScoredExplanation:
            return AttributedScoredExplanation(
                explanation=scored_explanation.explanation,
                score=scored_explanation.get_preferred_score(),
                dataset_name=explanation_dataset.split("/")[-1],
            )

        async def load_and_convert_explanations(
            explanation_dataset: str,
        ) -> list[AttributedScoredExplanation]:
            neuron_simulation_results = await load_neuron_explanations_async(
                explanation_dataset, request.layer_index, request.activation_index
            )
            if neuron_simulation_results is None:
                return []
            else:
                return [
                    convert_scored_explanation(scored_explanation, explanation_dataset)
                    for scored_explanation in neuron_simulation_results.scored_explanations
                    if scored_explanation.explanation is not None
                ]

        if not ((len(request.explanation_datasets) > 0) ^ (request.neuron_dataset is not None)):
            raise HTTPException(
                status_code=400,
                detail="Exactly one of explanation_datasets and neuron_dataset must be specified.",
            )

        if len(request.explanation_datasets) > 0:
            explanation_datasets = request.explanation_datasets
        else:
            assert request.neuron_dataset is not None  # Redundant assert; mypy needs this.
            neuron_dataset = resolve_neuron_dataset(request.neuron_dataset, request.dst)
            explanation_datasets = await get_all_explanation_datasets(neuron_dataset)

        tasks = [load_and_convert_explanations(dataset) for dataset in explanation_datasets]
        scored_explanation_lists = await asyncio.gather(*tasks)
        # Flatten the list of lists.
        return [item for sublist in scored_explanation_lists for item in sublist]

    @app.post("/neuron_record", response_model=NeuronRecordResponse, tags=["read"])
    async def neuron_record(request: NodeIdAndDatasets) -> NeuronRecordResponse:
        dataset_path, neuron_record = await load_neuron_from_datasets(request)
        top_activations, random_sample = convert_activation_records_to_token_and_activation_lists(
            [
                neuron_record.most_positive_activation_records,
                neuron_record.random_sample,
            ]
        )
        return NeuronRecordResponse(
            dataset=dataset_path,
            max_activation=neuron_record.max_activation,
            top_activations=top_activations,
            random_sample=random_sample,
        )

    @app.post("/attention_head_record", response_model=AttentionHeadRecordResponse, tags=["read"])
    async def attention_head_record(request: NodeIdAndDatasets) -> AttentionHeadRecordResponse:
        dataset_path, neuron_record = await load_neuron_from_datasets(request)
        (
            most_positive_token_sequences,
            random_token_sequences,
        ) = convert_activation_records_to_token_and_attention_activations_lists(
            [
                neuron_record.most_positive_activation_records,
                neuron_record.random_sample,
            ]
        )

        return AttentionHeadRecordResponse(
            dataset=dataset_path,
            max_attention_activation=neuron_record.max_activation,
            most_positive_token_sequences=most_positive_token_sequences,
            random_token_sequences=random_token_sequences,
        )

    @app.post(
        "/neuron_datasets_metadata", response_model=list[NeuronDatasetMetadata], tags=["read"]
    )
    def neuron_datasets_metadata() -> list[NeuronDatasetMetadata]:
        return get_all_neuron_dataset_metadata()


def flatten(list_of_lists: list[list[T]]) -> list[T]:
    return [item for sublist in list_of_lists for item in sublist]


def normalize_token_scalars(
    list_of_sequence_lists: list[list[list[TokenAndRawScalar]]],
) -> list[list[list[TokenAndScalar]]]:
    """The input ist a list of lists of lists of TokenAndRawScalar objects; the
    outer list is indexed by provenance, e.g. top or random token sequences;
    the middle list is indexed by token sequence; the inner list is indexed by token
    within a sequence.
    flatten() collapses the outermost level of nesting. We first flatten the whole thing,
    to compute the max activation across all tokens in all sequences of all provenances.
    We floor at 0, so if all activations are negative max_activation is 0.

    Then we step through each token in each sequence in each provenance, and divide by
    that max activation, and floor at 0, to get a normalized activation with ceiling at 1
    and floor at 0. This normalized_activation goes in a TokenAndScalar object, which
    is in a nested list structure parallel to the input structure."""
    indexed_by_token: list[TokenAndRawScalar] = flatten(flatten(list_of_sequence_lists))
    indexed_by_token = [
        TokenAndRawScalar(token=d.token, scalar=max(d.scalar, 0)) for d in indexed_by_token
    ]
    max_scalar = max([d.scalar for d in indexed_by_token]) or 0
    neuron_scale = create_scale_linear(max_scalar)

    return [
        [
            [
                TokenAndScalar(
                    token=d.token,
                    scalar=d.scalar,
                    normalized_scalar=neuron_scale(d.scalar),
                )
                for d in sequence
            ]
            for sequence in list_of_sequences
        ]
        for list_of_sequences in list_of_sequence_lists
    ]


def create_scale_linear(value: float) -> Callable[[float], float]:
    return lambda x: max((x - 0) / (value - 1e-5), 0)


def normalize_attention_token_scalars(
    list_of_sequence_lists: list[list[list[TokenAndRawAttentionScalars]]],
) -> list[list[list[TokenAndAttentionScalars]]]:
    """The high level setup is analogous to normalize_token_scalars, but using
    TokenAndRawAttentionScalars objects, which have a list of floats associated
    to each token. There are several kinds of normalizations applied:

    1. Normalizing all scalars in all provenances, in all sequences, in all tokens,
    all scalars within a token's scalars, to the same scale so that they are in
    the range [0, 1].
    2. Summarizing the lists of scalars per token in some way to get a single scalar
    per token, and then normalizing those scalars to the same scale, as described in 1.and
    a. one way to summarize is to compute a summary statistic on the "scalars in", which are
    the list of floats associated to each token.
    b. the other way is to compute a summary statistic on the "scalars out", which are the
    ith entry in each list of of floats that contains an ith entry.
    i. one summary statistic used is the "total", or sum of all scalars in or out
    ii. another summary statistic used is the "max", or maximum of all scalars in or out

    These normalized and summarized scalars are then stored in a TokenAndAttentionScalars object,
    which is returned in a nested list structure parallel to the input structure.
    """

    indexed_by_token_sequence = flatten(list_of_sequence_lists)
    indexed_by_token = flatten(indexed_by_token_sequence)
    indexed_by_token = [
        TokenAndRawAttentionScalars(token=d.token, scalars=[max(a, 0) for a in d.scalars])
        for d in indexed_by_token
    ]
    max_scalar = max([a for d in indexed_by_token for a in d.scalars]) or 0
    total_scalar_scale = create_scale_linear(max_scalar)

    def compute_summary_of_scalar_in(
        attn_token_sequence_list: list[TokenAndRawAttentionScalars],
        operation: Callable[[Sequence[float | None]], float],
    ) -> list[float]:
        return [operation(d.scalars) for d in attn_token_sequence_list]

    def _get_entry_if_available(scalars: Sequence[float], index: int) -> float | None:
        # if index is out of bounds, return None; otherwise return the entry at that index
        return scalars[index] if index < len(scalars) else None

    def _sum_non_none(scalars: Sequence[float | None]) -> float:
        # sum all non-None entries in scalars
        return sum([a for a in scalars if a is not None])

    def _max_non_none(scalars: Sequence[float | None]) -> float:
        # return the max of all non-None entries in scalars
        return max([a for a in scalars if a is not None])

    def compute_summary_of_scalar_out(
        attn_token_sequence_list: list[TokenAndRawAttentionScalars],
        operation: Callable[[Sequence[float | None]], float],
    ) -> list[float]:
        for i in range(len(attn_token_sequence_list)):
            # the attended to sequence length at token i is at most i+1
            assert len(attn_token_sequence_list[i].scalars) <= i + 1, (
                i,
                len(attn_token_sequence_list[i].scalars),
            )
        scalar_out = [
            operation(
                [
                    _get_entry_if_available(attn_token_sequence_list[i].scalars, j)
                    for i in range(j, len(attn_token_sequence_list))
                ]
            )
            for j in range(len(attn_token_sequence_list))
        ]
        return scalar_out

    def compute_total_scalar_in(
        attn_token_sequence_list: list[TokenAndRawAttentionScalars],
    ) -> list[float]:
        return compute_summary_of_scalar_in(attn_token_sequence_list, _sum_non_none)

    def compute_max_scalar_in(
        attn_token_sequence_list: list[TokenAndRawAttentionScalars],
    ) -> list[float]:
        return compute_summary_of_scalar_in(attn_token_sequence_list, _max_non_none)

    def compute_total_scalar_out(
        attn_token_sequence_list: list[TokenAndRawAttentionScalars],
    ) -> list[float]:
        return compute_summary_of_scalar_out(attn_token_sequence_list, _sum_non_none)

    def compute_max_scalar_out(
        attn_token_sequence_list: list[TokenAndRawAttentionScalars],
    ) -> list[float]:
        return compute_summary_of_scalar_out(attn_token_sequence_list, _max_non_none)

    def compute_scalar_summary_and_scale(
        compute_scalar_summary_function: Callable[[list[TokenAndRawAttentionScalars]], list[float]],
        list_of_sequence_lists: list[list[list[TokenAndRawAttentionScalars]]],
    ) -> tuple[list[list[list[float]]], Callable[[float], float]]:
        scalar_indexed_by_token_sequence_list: list[list[list[float]]] = [
            [compute_scalar_summary_function(sequence) for sequence in sequence_list]
            for sequence_list in list_of_sequence_lists
        ]
        scalar_indexed_by_token: list[float] = flatten(
            flatten(scalar_indexed_by_token_sequence_list)
        )
        scalar_indexed_by_token = [max(a, 0) for a in scalar_indexed_by_token]
        scalar_scale: Callable[[float], float] = create_scale_linear(
            max(scalar_indexed_by_token) or 0
        )
        return scalar_indexed_by_token_sequence_list, scalar_scale

    (
        total_scalar_in_indexed_by_token_sequence_list,
        total_scalar_in_scale,
    ) = compute_scalar_summary_and_scale(compute_total_scalar_in, list_of_sequence_lists)
    (
        max_scalar_in_indexed_by_token_sequence_list,
        max_scalar_in_scale,
    ) = compute_scalar_summary_and_scale(compute_max_scalar_in, list_of_sequence_lists)
    (
        total_scalar_out_indexed_by_token_sequence_list,
        total_scalar_out_scale,
    ) = compute_scalar_summary_and_scale(compute_total_scalar_out, list_of_sequence_lists)
    (
        max_scalar_out_indexed_by_token_sequence_list,
        max_scalar_out_scale,
    ) = compute_scalar_summary_and_scale(compute_max_scalar_out, list_of_sequence_lists)

    return [
        [
            [
                TokenAndAttentionScalars(
                    token=d.token,
                    scalars=d.scalars,
                    normalized_scalars=[total_scalar_scale(a) for a in d.scalars],
                    total_scalar_in=total_scalar_in_indexed_by_token_sequence_list[seq_list_idx][
                        seq_idx
                    ][token_idx],
                    normalized_total_scalar_in=total_scalar_in_scale(
                        total_scalar_in_indexed_by_token_sequence_list[seq_list_idx][seq_idx][
                            token_idx
                        ]
                    ),
                    max_scalar_in=max_scalar_in_indexed_by_token_sequence_list[seq_list_idx][
                        seq_idx
                    ][token_idx],
                    normalized_max_scalar_in=max_scalar_in_scale(
                        max_scalar_in_indexed_by_token_sequence_list[seq_list_idx][seq_idx][
                            token_idx
                        ]
                    ),
                    total_scalar_out=total_scalar_out_indexed_by_token_sequence_list[seq_list_idx][
                        seq_idx
                    ][token_idx],
                    normalized_total_scalar_out=total_scalar_out_scale(
                        total_scalar_out_indexed_by_token_sequence_list[seq_list_idx][seq_idx][
                            token_idx
                        ]
                    ),
                    max_scalar_out=max_scalar_out_indexed_by_token_sequence_list[seq_list_idx][
                        seq_idx
                    ][token_idx],
                    normalized_max_scalar_out=max_scalar_out_scale(
                        max_scalar_out_indexed_by_token_sequence_list[seq_list_idx][seq_idx][
                            token_idx
                        ]
                    ),
                )
                for token_idx, d in enumerate(sequence)
            ]
            for seq_idx, sequence in enumerate(list_of_sequences)
        ]
        for seq_list_idx, list_of_sequences in enumerate(list_of_sequence_lists)
    ]
