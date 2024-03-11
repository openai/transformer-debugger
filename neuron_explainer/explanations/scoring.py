from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Sequence

import numpy as np

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.calibrated_simulator import (
    CalibratedNeuronSimulator,
    LinearCalibratedNeuronSimulator,
    UncalibratedNeuronSimulator,
)
from neuron_explainer.explanations.explanations import (
    ScoredSequenceSimulation,
    ScoredSimulation,
    SequenceSimulation,
)
from neuron_explainer.explanations.simulator import (
    ExplanationNeuronSimulator,
    LogprobFreeExplanationTokenSimulator,
    NeuronSimulator,
)


def flatten_list(list_of_lists: Sequence[Sequence[Any]]) -> list[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def correlation_score(
    real_activations: Sequence[float] | np.ndarray,
    predicted_activations: Sequence[float] | np.ndarray,
) -> float:
    score = np.corrcoef(real_activations, predicted_activations)[0, 1]
    if np.isnan(score):
        return 0.0
    return score


def score_from_simulation(
    real_activations: ActivationRecord,
    simulation: SequenceSimulation,
    score_function: Callable[[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray], float],
) -> float:
    return score_function(real_activations.activations, simulation.expected_activations)


def rsquared_score_from_sequences(
    real_activations: Sequence[float] | np.ndarray,
    predicted_activations: Sequence[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(np.square(np.array(real_activations) - np.array(predicted_activations)))
        / np.mean(np.square(np.array(real_activations)))
    )


def absolute_dev_explained_score_from_sequences(
    real_activations: Sequence[float] | np.ndarray,
    predicted_activations: Sequence[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(np.abs(np.array(real_activations) - np.array(predicted_activations)))
        / np.mean(np.abs(np.array(real_activations)))
    )


async def make_uncalibrated_explanation_simulator(
    explanation: str,
    client: ApiClient,
    **kwargs: Any,
) -> CalibratedNeuronSimulator:
    """Make a simulator that doesn't apply any calibration."""
    simulator = LogprobFreeExplanationTokenSimulator(client, explanation, **kwargs)
    calibrated_simulator = UncalibratedNeuronSimulator(simulator)
    return calibrated_simulator


async def make_explanation_simulator(
    explanation: str,
    calibration_activation_records: Sequence[ActivationRecord],
    client: ApiClient,
    calibrated_simulator_class: type[CalibratedNeuronSimulator] = LinearCalibratedNeuronSimulator,
    **kwargs: Any,
) -> CalibratedNeuronSimulator:
    """
    Make a simulator that uses an explanation to predict activations and calibrates it on the given
    activation records.
    """
    simulator = ExplanationNeuronSimulator(client, explanation, **kwargs)
    calibrated_simulator = calibrated_simulator_class(simulator)
    await calibrated_simulator.calibrate(calibration_activation_records)
    return calibrated_simulator


async def _simulate_and_score_sequence(
    simulator: NeuronSimulator, activations: ActivationRecord
) -> ScoredSequenceSimulation:
    """Score an explanation of a neuron by how well it predicts activations on a sentence."""
    sequence_simulation = await simulator.simulate(activations.tokens)
    logging.debug(sequence_simulation)
    rsquared_score = score_from_simulation(
        activations, sequence_simulation, rsquared_score_from_sequences
    )
    absolute_dev_explained_score = score_from_simulation(
        activations, sequence_simulation, absolute_dev_explained_score_from_sequences
    )
    scored_sequence_simulation = ScoredSequenceSimulation(
        sequence_simulation=sequence_simulation,
        true_activations=activations.activations,
        ev_correlation_score=score_from_simulation(
            activations, sequence_simulation, correlation_score
        ),
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )
    return scored_sequence_simulation


def aggregate_scored_sequence_simulations(
    scored_sequence_simulations: list[ScoredSequenceSimulation],
) -> ScoredSimulation:
    """
    Aggregate a list of scored sequence simulations. The logic for doing this is non-trivial for EV
    scores, since we want to calculate the correlation over all activations from all sequences at
    once rather than simply averaging per-sequence correlations.
    """
    all_true_activations: list[float] = []
    all_expected_values: list[float] = []
    for scored_sequence_simulation in scored_sequence_simulations:
        all_true_activations.extend(scored_sequence_simulation.true_activations or [])
        all_expected_values.extend(
            scored_sequence_simulation.sequence_simulation.expected_activations
        )
    ev_correlation_score = (
        correlation_score(all_true_activations, all_expected_values)
        if len(all_true_activations) > 0
        else None
    )
    rsquared_score = rsquared_score_from_sequences(all_true_activations, all_expected_values)
    absolute_dev_explained_score = absolute_dev_explained_score_from_sequences(
        all_true_activations, all_expected_values
    )

    return ScoredSimulation(
        scored_sequence_simulations=scored_sequence_simulations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )


async def simulate_and_score(
    simulator: NeuronSimulator,
    activation_records: Sequence[ActivationRecord],
) -> ScoredSimulation:
    """
    Score an explanation of a neuron by how well it predicts activations on the given text
    sequences.
    """
    scored_sequence_simulations = await asyncio.gather(
        *[
            _simulate_and_score_sequence(
                simulator,
                activation_record,
            )
            for activation_record in activation_records
        ]
    )
    return aggregate_scored_sequence_simulations(scored_sequence_simulations)


async def make_simulator_and_score(
    make_simulator: Coroutine[None, None, NeuronSimulator],
    activation_records: Sequence[ActivationRecord],
) -> ScoredSimulation:
    """Chain together creating the simulator and using it to score activation records."""
    simulator = await make_simulator
    return await simulate_and_score(simulator, activation_records)
