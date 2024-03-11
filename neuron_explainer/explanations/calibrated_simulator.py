"""
Code for calibrating simulations of neuron behavior. Calibration refers to a process of mapping from
a space of predicted activation values (e.g. [0, 10]) to the real activation distribution for a
neuron.

See http://go/neuron_explanation_methodology for description of calibration step. Necessary for
simulating neurons in the context of ablate-to-simulation, but can be skipped when using correlation
scoring. (Calibration may still improve quality for scoring, at least for non-linear calibration
methods.)
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Sequence

import numpy as np
from sklearn import linear_model

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.explanations import ActivationScale
from neuron_explainer.explanations.simulator import NeuronSimulator, SequenceSimulation


class CalibratedNeuronSimulator(NeuronSimulator):
    """
    Wrap a NeuronSimulator and calibrate it to map from the predicted activation space to the
    actual neuron activation space.
    """

    def __init__(self, uncalibrated_simulator: NeuronSimulator):
        self.uncalibrated_simulator = uncalibrated_simulator

    @classmethod
    async def create(
        cls,
        uncalibrated_simulator: NeuronSimulator,
        calibration_activation_records: Sequence[ActivationRecord],
    ) -> CalibratedNeuronSimulator:
        """
        Create and calibrate a calibrated simulator (so initialization and calibration can be done
        in one call).
        """
        calibrated_simulator = cls(uncalibrated_simulator)
        await calibrated_simulator.calibrate(calibration_activation_records)
        return calibrated_simulator

    async def calibrate(self, calibration_activation_records: Sequence[ActivationRecord]) -> None:
        """
        Determine parameters to map from the predicted activation space to the real neuron
        activation space, based on a calibration set.

        Use when simulated sequences haven't already been produced on the calibration set.
        """
        simulations = await asyncio.gather(
            *[
                self.uncalibrated_simulator.simulate(activations.tokens)
                for activations in calibration_activation_records
            ]
        )
        self.calibrate_from_simulations(calibration_activation_records, simulations)

    def calibrate_from_simulations(
        self,
        calibration_activation_records: Sequence[ActivationRecord],
        simulations: Sequence[SequenceSimulation],
    ) -> None:
        """
        Determine parameters to map from the predicted activation space to the real neuron
        activation space, based on a calibration set.

        Use when simulated sequences have already been produced on the calibration set.
        """
        flattened_activations = []
        flattened_simulated_activations: list[float] = []
        for activations, simulation in zip(calibration_activation_records, simulations):
            flattened_activations.extend(activations.activations)
            flattened_simulated_activations.extend(simulation.expected_activations)
        self._calibrate_from_flattened_activations(
            np.array(flattened_activations), np.array(flattened_simulated_activations)
        )

    @abstractmethod
    def _calibrate_from_flattened_activations(
        self,
        true_activations: np.ndarray,
        uncalibrated_activations: np.ndarray,
    ) -> None:
        """
        Determine parameters to map from the predicted activation space to the real neuron
        activation space, based on a calibration set.

        Take numpy arrays of all true activations and all uncalibrated activations on the
        calibration set over all sequences.
        """

    @abstractmethod
    def apply_calibration(self, values: Sequence[float]) -> list[float]:
        """Apply the learned calibration to a sequence of values."""

    async def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        uncalibrated_seq_simulation = await self.uncalibrated_simulator.simulate(tokens)
        calibrated_activations = self.apply_calibration(
            uncalibrated_seq_simulation.expected_activations
        )
        calibrated_distribution_values = [
            self.apply_calibration(dv) for dv in uncalibrated_seq_simulation.distribution_values
        ]
        return SequenceSimulation(
            activation_scale=ActivationScale.NEURON_ACTIVATIONS,
            tokens=uncalibrated_seq_simulation.tokens,
            expected_activations=calibrated_activations,
            distribution_values=calibrated_distribution_values,
            distribution_probabilities=uncalibrated_seq_simulation.distribution_probabilities,
            uncalibrated_simulation=uncalibrated_seq_simulation,
        )


class UncalibratedNeuronSimulator(CalibratedNeuronSimulator):
    """Pass through the activations without trying to calibrate."""

    def __init__(self, uncalibrated_simulator: NeuronSimulator):
        super().__init__(uncalibrated_simulator)

    async def calibrate(self, calibration_activation_records: Sequence[ActivationRecord]) -> None:
        pass

    def _calibrate_from_flattened_activations(
        self,
        true_activations: np.ndarray,
        uncalibrated_activations: np.ndarray,
    ) -> None:
        pass

    def apply_calibration(self, values: Sequence[float]) -> list[float]:
        return values if isinstance(values, list) else list(values)


class LinearCalibratedNeuronSimulator(CalibratedNeuronSimulator):
    """Find a linear mapping from uncalibrated activations to true activations.

    Should not change ev_correlation_score because it is invariant to linear transformations.
    """

    def __init__(self, uncalibrated_simulator: NeuronSimulator):
        super().__init__(uncalibrated_simulator)
        self._regression: linear_model.LinearRegression | None = None

    def _calibrate_from_flattened_activations(
        self,
        true_activations: np.ndarray,
        uncalibrated_activations: np.ndarray,
    ) -> None:
        self._regression = linear_model.LinearRegression()
        self._regression.fit(uncalibrated_activations.reshape(-1, 1), true_activations)

    def apply_calibration(self, values: Sequence[float]) -> list[float]:
        if self._regression is None:
            raise ValueError("Must call calibrate() before apply_calibration")
        if len(values) == 0:
            return []
        return self._regression.predict(np.reshape(np.array(values), (-1, 1))).tolist()


class PercentileMatchingCalibratedNeuronSimulator(CalibratedNeuronSimulator):
    """
    Map the nth percentile of the uncalibrated activations to the nth percentile of the true
    activations for all n.

    This will match the distribution of true activations on the calibration set, but will be
    overconfident outside of the calibration set.
    """

    def __init__(self, uncalibrated_simulator: NeuronSimulator):
        super().__init__(uncalibrated_simulator)
        self._uncalibrated_activations: np.ndarray | None = None
        self._true_activations: np.ndarray | None = None

    def _calibrate_from_flattened_activations(
        self,
        true_activations: np.ndarray,
        uncalibrated_activations: np.ndarray,
    ) -> None:
        self._uncalibrated_activations = np.sort(uncalibrated_activations)
        self._true_activations = np.sort(true_activations)

    def apply_calibration(self, values: Sequence[float]) -> list[float]:
        if self._true_activations is None or self._uncalibrated_activations is None:
            raise ValueError("Must call calibrate() before apply_calibration")
        if len(values) == 0:
            return []
        return np.interp(
            np.array(values), self._uncalibrated_activations, self._true_activations
        ).tolist()
