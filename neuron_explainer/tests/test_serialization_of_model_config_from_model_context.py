import json

from neuron_explainer.models.model_context import StandardModelContext


def test_standard_model_context(standard_model_context: StandardModelContext) -> None:
    json.dumps(standard_model_context.get_model_config_as_dict())
