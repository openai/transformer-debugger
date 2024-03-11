# This file defines fixtures for model tests, with a focus on expensive objects that are used across
# multiple test files. Fixtures are created once per session (i.e. `pytest` invocation), and are
# available to and reused across all test cases in the session. Fixtures are evaluated lazily.
# The filename uses the pytest convention.

import pytest

from neuron_explainer.activations.derived_scalars import DerivedScalarType
from neuron_explainer.activations.derived_scalars.tests.utils import get_autoencoder_test_path
from neuron_explainer.models.autoencoder_context import AutoencoderConfig, AutoencoderContext
from neuron_explainer.models.model_context import StandardModelContext, get_default_device

AUTOENCODER_TEST_DST = DerivedScalarType.MLP_POST_ACT
AUTOENCODER_TEST_PATH = get_autoencoder_test_path(AUTOENCODER_TEST_DST)


@pytest.fixture(scope="session")
def standard_model_context() -> StandardModelContext:
    standard_model_context = StandardModelContext.from_model_type(
        "gpt2-small", device=get_default_device()
    )
    assert isinstance(standard_model_context, StandardModelContext)
    return standard_model_context


@pytest.fixture(scope="session")
def standard_autoencoder_context(
    standard_model_context: StandardModelContext,
) -> AutoencoderContext:
    autoencoder_config = AutoencoderConfig(
        dst=AUTOENCODER_TEST_DST,
        autoencoder_path_by_layer_index={
            layer_index: AUTOENCODER_TEST_PATH
            for layer_index in range(standard_model_context.n_layers)
        },
    )
    return AutoencoderContext(
        autoencoder_config=autoencoder_config,
        device=standard_model_context.device,
    )
