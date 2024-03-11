# This file supports translation between universal representation and inference engine-specific
# representations. The current codebase only supports one inference engine, but that may change in
# the future.

from enum import Enum

from neuron_explainer.models.model_component_registry import ActivationLocationType, LayerIndex

# TODO: Consider using a stronger type here.
StandardModelHookLocationType = str
HookLocationType = StandardModelHookLocationType


class InferenceEngineType(str, Enum):
    STANDARD = "standard"


_standard_model_hook_location_type_by_activation_location_type: dict[
    ActivationLocationType, StandardModelHookLocationType
] = {
    ActivationLocationType.RESID_POST_EMBEDDING: "resid/post_emb",
    ActivationLocationType.RESID_POST_MLP: "resid/post_mlp",
    ActivationLocationType.MLP_PRE_ACT: "mlp/pre_act",
    ActivationLocationType.MLP_POST_ACT: "mlp/post_act",
    ActivationLocationType.ATTN_QUERY: "attn/q",
    ActivationLocationType.ATTN_KEY: "attn/k",
    ActivationLocationType.ATTN_VALUE: "attn/v",
    ActivationLocationType.ATTN_QK_LOGITS: "attn/qk_logits",
    ActivationLocationType.ATTN_QK_PROBS: "attn/qk_probs",
    ActivationLocationType.ATTN_WEIGHTED_SUM_OF_VALUES: "attn/v_out",
    ActivationLocationType.RESID_DELTA_ATTN: "resid/delta_attn",
    ActivationLocationType.RESID_DELTA_MLP: "resid/delta_mlp",
    ActivationLocationType.RESID_POST_ATTN: "resid/post_attn",
    ActivationLocationType.RESID_POST_MLP: "resid/post_mlp",
    ActivationLocationType.LOGITS: "logits",
    ActivationLocationType.RESID_FINAL_LAYER_NORM_SCALE: "resid/ln_f/scale",
    ActivationLocationType.ATTN_INPUT_LAYER_NORM_SCALE: "resid/ln_attn/scale",
    ActivationLocationType.MLP_INPUT_LAYER_NORM_SCALE: "resid/ln_mlp/scale",
}


_hook_location_type_by_activation_location_type_by_inference_engine_type: dict[
    InferenceEngineType, dict[ActivationLocationType, HookLocationType]
] = {
    InferenceEngineType.STANDARD: _standard_model_hook_location_type_by_activation_location_type,
}


_activation_location_type_by_hook_location_type_by_inference_engine_type: dict[
    InferenceEngineType, dict[HookLocationType, ActivationLocationType]
] = {
    inference_engine_type: {v: k for k, v in hook_location_type_by_activation_location_type.items()}
    for inference_engine_type, hook_location_type_by_activation_location_type in _hook_location_type_by_activation_location_type_by_inference_engine_type.items()
}


standard_model_activation_location_types: set[ActivationLocationType] = set(
    _hook_location_type_by_activation_location_type_by_inference_engine_type[
        InferenceEngineType.STANDARD
    ].keys()
)


def get_hook_location_type_for_activation_location_type(
    activation_location_type: ActivationLocationType, inference_engine_type: InferenceEngineType
) -> HookLocationType:
    assert (
        inference_engine_type
        in _hook_location_type_by_activation_location_type_by_inference_engine_type
    ), f"Unknown inference_engine_type {inference_engine_type}"
    return _hook_location_type_by_activation_location_type_by_inference_engine_type[
        inference_engine_type
    ][activation_location_type]


def get_activation_location_type_for_hook_location_type(
    hook_location_type: HookLocationType, inference_engine_type: InferenceEngineType
) -> ActivationLocationType:
    assert (
        inference_engine_type
        in _activation_location_type_by_hook_location_type_by_inference_engine_type
    ), f"Unknown inference_engine_type {inference_engine_type}"
    return _activation_location_type_by_hook_location_type_by_inference_engine_type[
        inference_engine_type
    ][hook_location_type]


def parse_standard_location_str(location: str) -> tuple[str, LayerIndex]:
    """
    Our transformer implementation uses location strings like "7/mlp/post_act" or "resid/post_emb"
    to capture both the location type and the layer index (if applicable). This function parses
    those strings, returning the location type and layer index. The location type is specific to our
    transformer implementation: call get_activation_location_type_for_hook_location_type to convert
    it into an ActivationLocationType.
    """
    if location[0].isdigit():
        # Example: 7/mlp/post_act
        return location[location.find("/") + 1 :], int(location.split("/")[0])
    else:
        # Example: resid/post_emb
        return location, None
