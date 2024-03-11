import time
from typing import Any, Callable

import torch

from neuron_explainer.models.inference_engine_type_registry import InferenceEngineType
from neuron_explainer.models.model_component_registry import WeightLocationType
from neuron_explainer.models.model_context import ModelContext


def assert_all_eq(
    lst: list[Any],
    eq_fn: Callable[[Any, Any], bool] = lambda x, y: x == y,
    weight_location_type: WeightLocationType | None = None,
) -> Any:
    for i in range(1, len(lst)):
        assert eq_fn(lst[i], lst[0]), f"{lst[i]} != {lst[0]}; {weight_location_type=}; {i=}"
    return lst[0]


def test_model_context_weights() -> None:
    for model_name in ["gpt2-small"]:
        contexts = []
        standard_model_context = ModelContext.from_model_type(
            model_name,
            inference_engine_type=InferenceEngineType.STANDARD,
            device="cpu",
        )
        contexts.append(("standard", standard_model_context))

        standard_model_context_with_model = ModelContext.from_model_type(
            model_name,
            inference_engine_type=InferenceEngineType.STANDARD,
            device="cpu",
        )
        standard_model_context_with_model.get_or_create_model(simplify=False)  # type: ignore
        contexts.append(("standard_cached", standard_model_context_with_model))

        for weight_location_type in WeightLocationType:
            if not weight_location_type.has_no_layers:
                # just test layer 0 for now
                layer_index: int | None = 0
            else:
                layer_index = None

            weights = []
            for ctx_name, ctx in contexts:
                try:
                    t = time.time()
                    # Convert all weights to float32, since different contexts may use different
                    # dtypes by default. torch.allclose requires dtypes to match.
                    weight = ctx.get_weight(weight_location_type, layer_index).to(torch.float32)
                    print(f"{ctx_name} {weight.shape=} loaded in {time.time() - t:.2f}s")
                    weights.append(weight)
                except NotImplementedError:
                    print(f"{weight_location_type} not implemented in {ctx_name} context")

            if len(weights):
                assert_all_eq(
                    [weight.shape for weight in weights], lambda x, y: x == y, weight_location_type
                )
                assert_all_eq(
                    list(weights),
                    lambda x, y: torch.allclose(x, y, atol=1e-5, rtol=1e-3),
                    weight_location_type,
                )
            else:
                print(f"no weights found for {weight_location_type}")
