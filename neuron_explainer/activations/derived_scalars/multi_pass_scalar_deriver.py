"""
MultiPassScalarDerivers extend the functionality of ScalarDerivers by specifying how to derive a
scalar from some combination of the activations in multiple prompts.

The pairs of identical interfaces are:

ScalarDeriver:MultiPassScalarDeriver
ScalarSource:MultiPassScalarSource
RawActivationStore:MultiPassRawActivationStore

Both sets of objects can be used in populating a DerivedScalarStore. The intention is that it should
be possible to swap
derived_scalar_store = DerivedScalarStore.derive_from_raw(
    multi_pass_raw_activation_store,
    multi_pass_scalar_derivers,
)
for
batched_derived_scalar_store = [
    DerivedScalarStore.derive_from_raw(
        raw_activation_store,
        scalar_derivers,
    )
    for scalar_derivers, raw_activation_store
    in zip(batched_scalar_derivers, batched_raw_activation_store)
]
in order to compute derived scalars combining activations across multiple prompts in a batch.

Probable TODO: make an ABC, from which both ScalarDeriver and MultiPassScalarDeriver inherit
"""

from abc import ABC, abstractmethod
from enum import Enum

from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.derived_scalar_store import RawActivationStore
from neuron_explainer.activations.derived_scalars.locations import LayerIndexer, StaticLayerIndexer
from neuron_explainer.activations.derived_scalars.scalar_deriver import (
    ActivationsAndMetadata,
    DerivedScalarTypeAndPassType,
    ScalarDeriver,
    ScalarSource,
)
from neuron_explainer.models.model_component_registry import (
    ActivationLocationType,
    ActivationLocationTypeAndPassType,
    LayerIndex,
    LocationWithinLayer,
    PassType,
)


class PromptId(Enum):
    MAIN = "main"
    BASELINE = "baseline"


class PromptCombo(Enum):
    MAIN = "main"
    BASELINE = "baseline"
    SUBTRACT_BASELINE = "subtract_baseline"

    @property
    def required_prompt_ids(self) -> tuple[PromptId, ...]:
        match self:
            case PromptCombo.MAIN:
                return (PromptId.MAIN,)
            case PromptCombo.BASELINE:
                return (PromptId.BASELINE,)
            case PromptCombo.SUBTRACT_BASELINE:
                return (PromptId.MAIN, PromptId.BASELINE)
            case _:
                raise NotImplementedError

    def compute(
        self, activations_by_prompt_id: dict[PromptId, ActivationsAndMetadata]
    ) -> ActivationsAndMetadata:
        match self:
            case PromptCombo.MAIN:
                assert len(activations_by_prompt_id) == 1
                main = activations_by_prompt_id.pop(PromptId.MAIN)
                return main
            case PromptCombo.BASELINE:
                assert len(activations_by_prompt_id) == 1
                baseline = activations_by_prompt_id.pop(PromptId.BASELINE)
                return baseline
            case PromptCombo.SUBTRACT_BASELINE:
                main = activations_by_prompt_id.pop(PromptId.MAIN)
                baseline = activations_by_prompt_id.pop(PromptId.BASELINE)
                assert len(activations_by_prompt_id) == 0
                return main - baseline
            case _:
                raise NotImplementedError

    def derive_from_raw(
        self,
        multi_pass_raw_activation_store: "MultiPassRawActivationStore",
        scalar_source: ScalarSource,
        desired_layer_indices: (
            list[LayerIndex] | None
        ),  # indicates layer indices to keep; None indicates keep all
    ) -> ActivationsAndMetadata:
        activations_by_prompt_id: dict[PromptId, ActivationsAndMetadata] = {}
        for prompt_id in self.required_prompt_ids:
            raw_activation_store = (
                multi_pass_raw_activation_store.raw_activation_store_by_prompt_id[prompt_id]
            )
            activations_by_prompt_id[prompt_id] = scalar_source.derive_from_raw(
                raw_activation_store, desired_layer_indices
            )
        return self.compute(activations_by_prompt_id)


class MultiPassScalarSource(ABC):
    pass_type: PassType
    layer_indexer: LayerIndexer

    @property
    @abstractmethod
    def exists_by_default(self) -> bool:
        # returns True if the activation is instantiated by default in a normal transformer forward pass
        # this is False for activations related to autoencoders or for non-trivial derived scalars
        pass

    @property
    @abstractmethod
    def dst(self) -> DerivedScalarType:
        pass

    @property
    def dst_and_pass_type(self) -> DerivedScalarTypeAndPassType:
        return DerivedScalarTypeAndPassType(
            self.dst,
            self.pass_type,
        )

    @property
    @abstractmethod
    def sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        pass

    @property
    @abstractmethod
    def location_within_layer(self) -> LocationWithinLayer | None:
        pass

    @property
    def layer_index(self) -> LayerIndex:
        """Convenience method to get the single layer index associated with this ScalarSource, if such a single layer index
        exists. Throws an error if it does not."""
        assert isinstance(self.layer_indexer, StaticLayerIndexer), (
            self.layer_indexer,
            "ScalarSource.layer_index should only be called for ScalarSource StaticLayerIndexer",
        )
        return self.layer_indexer.layer_index

    @abstractmethod
    def derive_from_raw(
        self,
        multi_pass_raw_activation_store: "MultiPassRawActivationStore",
        desired_layer_indices: (
            list[LayerIndex] | None
        ),  # indicates layer indices to keep; None indicates keep all
    ) -> ActivationsAndMetadata:
        """
        See scalar_deriver.ScalarSource.derive_from_raw for explanation.
        """
        pass


class SinglePromptComboScalarSource(MultiPassScalarSource):
    """
    A SinglePromptComboScalarSource can be computed using some function
    derived_scalar_A = f(derived_scalar_A_from_one_prompt[, derived_scalar_A_from_another_prompt, ...])
    This is distinct from a MixedScalarSource, which is computed using some function of derived
    scalars from SinglePromptComboScalarSources or other MixedScalarSources. For example, a
    MixedScalarSource might be computed using some function:
    derived_scalar_A = f(
        g(sub_derived_scalar_B_from_one_prompt[, sub_derived_scalar_B_from_another_prompt, ...]),
        h(sub_derived_scalar_C_from_one_prompt[, sub_derived_scalar_C_from_another_prompt, ...]),
    )
    """

    scalar_source: ScalarSource
    prompt_combo: PromptCombo

    def __init__(self, scalar_source: ScalarSource, prompt_combo: PromptCombo):
        self.scalar_source = scalar_source
        self.prompt_combo = prompt_combo

    @property
    def exists_by_default(self) -> bool:
        return self.scalar_source.exists_by_default

    @property
    def dst(self) -> DerivedScalarType:
        return self.scalar_source.dst

    @property
    def sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        return self.scalar_source.sub_activation_location_type_and_pass_types

    @property
    def location_within_layer(self) -> LocationWithinLayer | None:
        return self.scalar_source.location_within_layer

    def derive_from_raw(
        self,
        multi_pass_raw_activation_store: "MultiPassRawActivationStore",
        desired_layer_indices: (
            list[LayerIndex] | None
        ),  # indicates layer indices to keep; None indicates keep all
    ) -> ActivationsAndMetadata:
        return self.prompt_combo.derive_from_raw(
            multi_pass_raw_activation_store, self.scalar_source, desired_layer_indices
        )


class MixedScalarSource(MultiPassScalarSource):
    multi_pass_scalar_deriver: "MultiPassScalarDeriver"
    pass_type: PassType
    layer_indexer: LayerIndexer

    def __init__(
        self,
        multi_pass_scalar_deriver: "MultiPassScalarDeriver",
        pass_type: PassType,
        layer_indexer: LayerIndexer,
    ):
        self.multi_pass_scalar_deriver = multi_pass_scalar_deriver
        self.pass_type = pass_type
        self.layer_indexer = layer_indexer

    @property
    def exists_by_default(self) -> bool:
        return False

    @property
    def dst(self) -> DerivedScalarType:
        return self.multi_pass_scalar_deriver.dst

    @property
    def sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        return self.multi_pass_scalar_deriver.get_sub_activation_location_type_and_pass_types()

    @property
    def location_within_layer(self) -> LocationWithinLayer | None:
        return self.multi_pass_scalar_deriver.scalar_deriver.location_within_layer

    def derive_from_raw(
        self,
        multi_pass_raw_activation_store: "MultiPassRawActivationStore",
        desired_layer_indices: (
            list[LayerIndex] | None
        ),  # indicates layer indices to keep; None indicates keep all
    ) -> ActivationsAndMetadata:
        return self.multi_pass_scalar_deriver.derive_from_raw(
            multi_pass_raw_activation_store, self.pass_type
        ).apply_layer_indexer(self.layer_indexer, desired_layer_indices)


class MultiPassScalarDeriver:
    scalar_deriver: ScalarDeriver
    sub_scalar_sources: tuple[MultiPassScalarSource, ...]

    def __init__(
        self, scalar_deriver: ScalarDeriver, sub_scalar_sources: tuple[MultiPassScalarSource, ...]
    ):
        self.scalar_deriver = scalar_deriver
        self.sub_scalar_sources = sub_scalar_sources
        assert [
            sub_scalar_source.dst_and_pass_type for sub_scalar_source in sub_scalar_sources
        ] == list(scalar_deriver.get_sub_dst_and_pass_types())

    @classmethod
    def from_scalar_deriver_and_sub_prompt_combos(
        cls,
        scalar_deriver: ScalarDeriver,
        sub_prompt_combos: tuple[PromptCombo, ...],
    ) -> "MultiPassScalarDeriver":
        assert len(scalar_deriver.sub_scalar_sources) == len(sub_prompt_combos)
        sub_scalar_sources = tuple(
            [
                SinglePromptComboScalarSource(scalar_source, prompt_combo)
                for scalar_source, prompt_combo in zip(
                    scalar_deriver.sub_scalar_sources, sub_prompt_combos
                )
            ]
        )
        return cls(scalar_deriver, sub_scalar_sources)

    @property
    def dst(self) -> DerivedScalarType:
        return self.scalar_deriver.dst

    @property
    def derivable_pass_types(self) -> tuple[PassType, ...]:
        return self.scalar_deriver.derivable_pass_types

    def activations_and_metadata_calculate_derived_scalar_fn(
        self, activation_data_tuple: tuple[ActivationsAndMetadata, ...], pass_type: PassType
    ) -> ActivationsAndMetadata:
        return self.scalar_deriver.activations_and_metadata_calculate_derived_scalar_fn(
            activation_data_tuple, pass_type
        )

    def get_sub_activation_location_type_and_pass_types(
        self,
    ) -> tuple[ActivationLocationTypeAndPassType, ...]:
        return self.scalar_deriver.get_sub_activation_location_type_and_pass_types()

    def derive_from_raw(
        self,
        multi_pass_raw_activation_store: "MultiPassRawActivationStore",
        pass_type: PassType,
    ) -> ActivationsAndMetadata:
        activations_list = []
        desired_layer_indices = None
        for scalar_source in self.sub_scalar_sources:
            activations = scalar_source.derive_from_raw(
                multi_pass_raw_activation_store, desired_layer_indices
            )
            activations_list.append(activations)
            if len(activations_list) == 1:
                # match the layer_indices of the first activations_and_metadata object
                desired_layer_indices = list(activations_list[0].layer_indices)
        return self.activations_and_metadata_calculate_derived_scalar_fn(
            tuple(activations_list),
            pass_type,
        )


# TODO: Run PromptCombo.derive_from_raw(scalar_source, raw_activation_store) as a part of
# MultiPassScalarSource.derive_from_raw(raw_activation_store)


class MultiPassRawActivationStore:
    raw_activation_store_by_prompt_id: dict[PromptId, RawActivationStore]

    def get_activations_and_metadata(
        self,
        prompt_id: PromptId,
        activation_location_type: ActivationLocationType,
        pass_type: PassType,
    ) -> ActivationsAndMetadata:
        return self.raw_activation_store_by_prompt_id[prompt_id].get_activations_and_metadata(
            activation_location_type, pass_type
        )
