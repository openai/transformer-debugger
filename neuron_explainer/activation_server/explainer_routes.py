"""Routes / endpoints related to generating and scoring explanations."""

from __future__ import annotations

import os
import os.path as osp
from enum import Enum, unique
from typing import Any, TypeVar

from fastapi import FastAPI, HTTPException

from neuron_explainer.activation_server.explanation_datasets import (
    AZURE_EXPLANATION_DATASET_REGISTRY,
    get_local_cached_explanation_directory,
)
from neuron_explainer.activation_server.load_neurons import load_neuron_from_datasets
from neuron_explainer.activation_server.read_routes import NodeIdAndDatasets
from neuron_explainer.activations.activations import (
    ActivationRecordSliceParams,
    NeuronId,
    NeuronRecord,
)
from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.attention_head_scoring import AttentionHeadOneAtATimeScorer
from neuron_explainer.explanations.explainer import (
    AttentionHeadExplainer,
    NeuronExplainer,
    TokenActivationPairExplainer,
)
from neuron_explainer.explanations.explanations import (
    AttentionSimulationResults,
    NeuronSimulationResults,
    ScoredAttentionExplanation,
    ScoredExplanation,
)
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import (
    make_simulator_and_score,
    make_uncalibrated_explanation_simulator,
)
from neuron_explainer.fast_dataclasses.fast_dataclasses import dumps, loads
from neuron_explainer.file_utils import file_exists, read_single_async
from neuron_explainer.models.model_component_registry import NodeType
from neuron_explainer.pydantic import CamelCaseBaseModel, immutable

T = TypeVar("T", bound="BaseMethodId")


@unique
class BaseMethodId(str, Enum):
    @classmethod
    def from_string(cls: type[T], s: str) -> T:
        for method_id in cls:
            if method_id.value == s:
                return method_id
        raise ValueError(f"{s} is not a valid {cls.__name__}")


@unique
class NeuronExplainAndScoreMethodId(BaseMethodId):
    BASELINE = "baseline"


_NEURON_EXPLAINER_REGISTRY: dict[NeuronExplainAndScoreMethodId, NeuronExplainer] = {
    NeuronExplainAndScoreMethodId.BASELINE: TokenActivationPairExplainer(
        model_name="gpt-4o",
        cache=True,
        prompt_format=PromptFormat.CHAT_MESSAGES,
    ),
}


@unique
class AttentionExplainAndScoreMethodId(BaseMethodId):
    BASELINE = "baseline"


# Maybe in the future will split this into one for the explainer and one for the scorer
_ATTENTION_EXPLAINER_REGISTRY: dict[
    AttentionExplainAndScoreMethodId, tuple[AttentionHeadExplainer, AttentionHeadOneAtATimeScorer]
] = {
    AttentionExplainAndScoreMethodId.BASELINE: (
        AttentionHeadExplainer(
            model_name="gpt-4o",
            prompt_format=PromptFormat.CHAT_MESSAGES,
            repeat_strongly_attending_pairs=True,
        ),
        AttentionHeadOneAtATimeScorer(
            model_name="gpt-4o",
            prompt_format=PromptFormat.CHAT_MESSAGES,
        ),
    )
}


@immutable
class ExplanationResult(CamelCaseBaseModel):
    explanations: list[str]
    # TODO(sbills): Get consistent about "dataset" vs "dataset_path".
    dataset: str


@immutable
class ScoreRequest(NodeIdAndDatasets):
    explanation: str
    max_sequences: int | None = None


@immutable
class ScoreResult(CamelCaseBaseModel):
    score: float
    dataset_path: str


@unique
class ActivationCategory(str, Enum):
    NEURON = "neuron"
    ATTENTION_HEAD = "attention_head"


def define_explainer_routes(
    app: FastAPI,
    neuron_method_id: NeuronExplainAndScoreMethodId,
    attention_head_method_id: AttentionExplainAndScoreMethodId,
) -> None:
    simulation_client = ApiClient(
        model_name="gpt-4o",
        max_concurrent=5,
        cache=True,
    )
    neuron_explainer = _NEURON_EXPLAINER_REGISTRY[neuron_method_id]
    attention_head_explainer, attention_head_scorer = _ATTENTION_EXPLAINER_REGISTRY[
        attention_head_method_id
    ]

    def _map_dst_to_activation_category(
        dst: DerivedScalarType,
    ) -> ActivationCategory:
        if dst in [
            DerivedScalarType.MLP_POST_ACT,
            DerivedScalarType.MLP_AUTOENCODER_LATENT,
            DerivedScalarType.ONLINE_MLP_AUTOENCODER_LATENT,
            DerivedScalarType.AUTOENCODER_LATENT,
            DerivedScalarType.ONLINE_AUTOENCODER_LATENT,
        ]:
            return ActivationCategory.NEURON
        elif dst in [
            DerivedScalarType.ATTN_WRITE_NORM,
            DerivedScalarType.ATTENTION_AUTOENCODER_LATENT,
            DerivedScalarType.ONLINE_ATTENTION_AUTOENCODER_LATENT,
            DerivedScalarType.FLATTENED_ATTN_WRITE_TO_LATENT_SUMMED_OVER_HEADS,
        ]:
            return ActivationCategory.ATTENTION_HEAD
        else:
            raise HTTPException(status_code=422, detail=f"Unsupported derived scalar type {dst}")

    def _get_azure_explanation_path(request: NodeIdAndDatasets, dataset_path: str) -> str | None:
        if dataset_path in AZURE_EXPLANATION_DATASET_REGISTRY:
            expl_dir = AZURE_EXPLANATION_DATASET_REGISTRY[dataset_path]
            return osp.join(expl_dir, str(request.layer_index), f"{request.activation_index}.jsonl")
        return None

    def _get_local_cached_explanation_path(request: NodeIdAndDatasets, dataset_path: str) -> str:
        if request.dst.node_type == NodeType.ATTENTION_HEAD:
            method_id_str = str(attention_head_method_id)
        else:
            method_id_str = str(neuron_method_id)
        cache_dir = get_local_cached_explanation_directory(dataset_path)
        return osp.join(
            cache_dir,
            f"cache_{request.dst}_{method_id_str}",
            str(request.layer_index),
            f"{request.activation_index}.jsonl",
        )

    def _verify_cached_simulation_results(
        request: NodeIdAndDatasets,
        simulation_results: Any,
    ) -> None:
        """Verifies the type and id of the cached simulation results."""
        if not isinstance(simulation_results, NeuronSimulationResults) and not isinstance(
            simulation_results, AttentionSimulationResults
        ):
            raise HTTPException(
                status_code=422, detail=f"Unexpected type {type(simulation_results)} in cache"
            )

        elem_id = (
            simulation_results.neuron_id
            if isinstance(simulation_results, NeuronSimulationResults)
            else simulation_results.attention_head_id
        )
        if (
            elem_id.layer_index != request.layer_index
            or elem_id.neuron_index != request.activation_index
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Cache id mismatch: requested ({request.layer_index}, {request.activation_index}, cache contained ({elem_id.layer_index}, {elem_id.layer_index})",
            )

    def _merge_simulation_results(
        azure_simulation_results: NeuronSimulationResults | AttentionSimulationResults | None,
        local_simulation_results: NeuronSimulationResults | AttentionSimulationResults | None,
    ) -> NeuronSimulationResults | AttentionSimulationResults | None:
        """Merge scored explanations from the local cache and azure into a single NeuronSimulationResults
        or AttentionSimulationResults object."""
        if azure_simulation_results is None and local_simulation_results is None:
            return None

        if isinstance(azure_simulation_results, NeuronSimulationResults) or isinstance(
            local_simulation_results, NeuronSimulationResults
        ):
            assert (
                isinstance(azure_simulation_results, NeuronSimulationResults)
                or azure_simulation_results is None
            )
            assert (
                isinstance(local_simulation_results, NeuronSimulationResults)
                or local_simulation_results is None
            )
            unique_scored_explanations = {}
            if azure_simulation_results is not None:
                for scored_explanation in azure_simulation_results.scored_explanations:
                    unique_scored_explanations[scored_explanation.explanation] = scored_explanation
            if local_simulation_results is not None:
                for scored_explanation in local_simulation_results.scored_explanations:
                    unique_scored_explanations[scored_explanation.explanation] = scored_explanation
            return NeuronSimulationResults(
                neuron_id=(
                    azure_simulation_results.neuron_id
                    if azure_simulation_results is not None
                    else local_simulation_results.neuron_id  # type: ignore # mypy doesn't understand that both can't be None
                ),
                scored_explanations=list(unique_scored_explanations.values()),
            )
        else:
            assert (
                isinstance(azure_simulation_results, AttentionSimulationResults)
                or azure_simulation_results is None
            )
            assert (
                isinstance(local_simulation_results, AttentionSimulationResults)
                or local_simulation_results is None
            )
            unique_scored_attn_explanations = {}
            if azure_simulation_results is not None:
                for scored_attn_explanation in azure_simulation_results.scored_explanations:
                    unique_scored_attn_explanations[
                        scored_attn_explanation.explanation
                    ] = scored_attn_explanation
            if local_simulation_results is not None:
                for scored_attn_explanation in local_simulation_results.scored_explanations:
                    unique_scored_attn_explanations[
                        scored_attn_explanation.explanation
                    ] = scored_attn_explanation
            return AttentionSimulationResults(
                attention_head_id=(
                    azure_simulation_results.attention_head_id
                    if azure_simulation_results is not None
                    else local_simulation_results.attention_head_id  # type: ignore # mypy doesn't understand that both can't be None
                ),
                scored_explanations=list(unique_scored_attn_explanations.values()),
            )

    async def _check_disk_for_simulation_results(
        request: NodeIdAndDatasets, dataset_path: str
    ) -> NeuronSimulationResults | AttentionSimulationResults | None:
        """
        If the request is for scored explanations in one of the public sets on azure, return them if any exist.
        Include any scored explanations in the local cache. If there are no explanations in azure or the local
        cache, return None.
        """
        azure_path = _get_azure_explanation_path(request, dataset_path)
        cache_path = _get_local_cached_explanation_path(request, dataset_path)
        azure_simulation_results, local_simulation_results = None, None
        if azure_path is not None and file_exists(azure_path):
            azure_simulation_results = loads(
                await read_single_async(azure_path), backwards_compatible=False
            )
            _verify_cached_simulation_results(request, azure_simulation_results)
        if file_exists(cache_path):
            local_simulation_results = loads(
                await read_single_async(cache_path), backwards_compatible=False
            )
            _verify_cached_simulation_results(request, local_simulation_results)

        # Merge the results from azure and the local cache, deduplicating any scored explanations.
        # Thus we have a single object that contains all the scored explanations for the node.
        combined_simulation_results = _merge_simulation_results(
            azure_simulation_results, local_simulation_results
        )
        return combined_simulation_results

    async def _explain_neuron(neuron_record: NeuronRecord) -> list[str]:
        if neuron_record.max_activation < 0:
            raise HTTPException(status_code=422, detail="Neuron is not activated on the dataset")
        train_activation_records = neuron_record.train_activation_records(
            activation_record_slice_params=ActivationRecordSliceParams(n_examples_per_split=5)
        )
        return await neuron_explainer.generate_explanations(
            all_activations=train_activation_records,
            max_activation=neuron_record.max_activation,
        )

    # NeuronRecord contains attention head activation info (it's an outdated name)
    async def _explain_attention_head(attention_record: NeuronRecord) -> list[str]:
        train_activation_records = attention_record.train_activation_records(
            activation_record_slice_params=ActivationRecordSliceParams(n_examples_per_split=5)
        )
        return await attention_head_explainer.generate_explanations(
            all_activations=train_activation_records,
            max_tokens=50,
            num_top_pairs_to_display=5,
        )

    @app.post("/explain", response_model=ExplanationResult, tags=["explainer"])
    async def explain(request: NodeIdAndDatasets) -> ExplanationResult:
        dataset_path, neuron_record = await load_neuron_from_datasets(request)
        cached_simulation_results = await _check_disk_for_simulation_results(request, dataset_path)

        if cached_simulation_results is not None:
            explanations = [s.explanation for s in cached_simulation_results.scored_explanations]
        else:
            activation_category = _map_dst_to_activation_category(request.dst)
            if activation_category == ActivationCategory.ATTENTION_HEAD:
                explanations = await _explain_attention_head(neuron_record)
            elif activation_category == ActivationCategory.NEURON:
                explanations = await _explain_neuron(neuron_record)
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported activation category for explanation: {activation_category}",
                )
        return ExplanationResult(explanations=explanations, dataset=dataset_path)

    async def _score_neuron(
        cached_simulation_results: NeuronSimulationResults | None,
        neuron: NeuronRecord,
        request: ScoreRequest,
        max_sequences: int | None = None,
    ) -> tuple[float | None, NeuronSimulationResults]:
        """Score an explanation for a neuron. Add it to the cached set of simulation
        results, or create the simulation results object if the cache was empty."""
        if neuron.max_activation < 0:
            raise HTTPException(status_code=422, detail="Neuron is not activated on the dataset")
        valid_activation_records = neuron.valid_activation_records(
            activation_record_slice_params=ActivationRecordSliceParams(n_examples_per_split=5)
        )

        if max_sequences is not None:
            valid_activation_records = valid_activation_records[:max_sequences]

        scored_simulation = await make_simulator_and_score(
            make_uncalibrated_explanation_simulator(
                request.explanation,
                simulation_client,
                prompt_format=PromptFormat.CHAT_MESSAGES,
            ),
            valid_activation_records,
        )

        scored_explanation = ScoredExplanation(
            explanation=request.explanation,
            scored_simulation=scored_simulation,
        )

        if cached_simulation_results is not None:
            cached_simulation_results.scored_explanations.append(scored_explanation)
        else:
            cached_simulation_results = NeuronSimulationResults(
                neuron_id=NeuronId(
                    neuron_index=request.activation_index,
                    layer_index=request.layer_index,
                ),
                scored_explanations=[scored_explanation],
            )
        return scored_explanation.get_preferred_score(), cached_simulation_results

    async def _score_attention_head(
        cached_simulation_results: AttentionSimulationResults | None,
        attention_record: NeuronRecord,
        request: ScoreRequest,
        max_sequences: int | None = None,
    ) -> tuple[float | None, AttentionSimulationResults]:
        """Score an explanation for an attention head. Add it to the cached set of simulation
        results, or create the simulation results object if the cache was empty."""
        if attention_record.max_activation < 0:
            raise HTTPException(
                status_code=422, detail="Attention head is not activated on the dataset"
            )
        valid_activation_records = attention_record.valid_activation_records(
            activation_record_slice_params=ActivationRecordSliceParams(n_examples_per_split=5)
        )

        if max_sequences is not None:
            valid_activation_records = valid_activation_records[:max_sequences]

        scored_attention_simulation = await attention_head_scorer.score_explanation(
            activation_records=valid_activation_records,
            explanation=request.explanation,
            max_activation=attention_record.max_activation,
        )

        scored_explanation = ScoredAttentionExplanation(
            explanation=request.explanation,
            scored_attention_simulation=scored_attention_simulation,
        )

        if cached_simulation_results is not None:
            cached_simulation_results.scored_explanations.append(scored_explanation)
        else:
            cached_simulation_results = AttentionSimulationResults(
                attention_head_id=NeuronId(
                    neuron_index=request.activation_index,
                    layer_index=request.layer_index,
                ),
                scored_explanations=[scored_explanation],
            )
        return scored_explanation.get_preferred_score(), cached_simulation_results

    def _cache_simulation_results_locally(
        request: ScoreRequest,
        dataset_path: str,
        cached_simulation_results: NeuronSimulationResults | AttentionSimulationResults,
    ) -> None:
        # Overwrite the cache with the updated set of simulation results.
        # Always cache locally because we can't write to the public azure bucket.
        cache_path = _get_local_cached_explanation_path(request, dataset_path)
        # Create the directories if they don't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(dumps(cached_simulation_results))

    @app.post("/score", response_model=ScoreResult, tags=["explainer"])
    async def score(request: ScoreRequest) -> ScoreResult:
        dataset_path, neuron_record = await load_neuron_from_datasets(request)
        cached_simulation_results = await _check_disk_for_simulation_results(request, dataset_path)

        # Cache hit: return the score for the matching explanation.
        if cached_simulation_results is not None and any(
            s.explanation == request.explanation
            for s in cached_simulation_results.scored_explanations
        ):
            score = [
                s.get_preferred_score()
                for s in cached_simulation_results.scored_explanations
                if s.explanation == request.explanation
            ][0]
            if score is None:
                raise HTTPException(status_code=500, detail="Score is unexpectedly undefined")
            return ScoreResult(score=score, dataset_path=dataset_path)

        # Cache miss: compute the score for the requested explanation.
        activation_category = _map_dst_to_activation_category(request.dst)
        if activation_category == ActivationCategory.ATTENTION_HEAD:
            assert cached_simulation_results is None or isinstance(
                cached_simulation_results, AttentionSimulationResults
            )
            # Score and update the cache storage object.
            score, cached_simulation_results = await _score_attention_head(
                cached_simulation_results,
                neuron_record,
                request,
                max_sequences=request.max_sequences,
            )
        elif activation_category == ActivationCategory.NEURON:
            assert cached_simulation_results is None or isinstance(
                cached_simulation_results, NeuronSimulationResults
            )
            # Score and update the cache storage object.
            score, cached_simulation_results = await _score_neuron(
                cached_simulation_results,
                neuron_record,
                request,
                max_sequences=request.max_sequences,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported activation category for scoring: {activation_category}",
            )

        if score is None:
            raise HTTPException(status_code=500, detail="Score is unexpectedly undefined")

        _cache_simulation_results_locally(request, dataset_path, cached_simulation_results)

        return ScoreResult(score=score, dataset_path=dataset_path)
