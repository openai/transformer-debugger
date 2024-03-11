import functools

import numpy as np
import pytest
import torch

from neuron_explainer.models import Transformer
from neuron_explainer.models.transformer import (
    causal_attn_mask,
    prep_input_and_pad,
    prep_pos_from_pad_and_prev_lens,
)

# models for testing on
REFERENCE_MODELS = ["gpt2/small"]
# uncomment this for more extensive testing
# REFERENCE_MODELS = ["gpt2/small", "gpt2/medium", "gpt2/large", "gpt2/xl"]

# if we run on a device with a GPU, let's test determinism on it!
REFERENCE_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

# convenience function for testing
assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


@functools.cache
def get_test_model(model, device):
    return Transformer.load(model, device=device)


# ======
# tests
# ======


def test_attention_masks():
    """Correctness tests for the attention mask functions."""

    # ======================
    # test causal_attn_mask
    # ======================
    M_ref = torch.BoolTensor(
        [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    )
    M = causal_attn_mask(size=5, device="cpu")

    assert_equal(M_ref, M, msg="causal_attn_mask produced incorrect result")


def test_sample_utilities():
    """Correctness tests for functions that enable batched sampling."""

    # ========================
    # test prep_input_and_pad
    # ========================
    # make a set of test prompts of unequal lengths
    test_inputs = [
        [1, 100, 50, 47],  # prompt 0
        [1298, 618, 952, 223, 4, 42],  # prompt 1
        [31],  # prompt 2
    ]
    X_ref = torch.LongTensor(
        [
            [0, 0, 1, 100, 50, 47],
            [1298, 618, 952, 223, 4, 42],
            [0, 0, 0, 0, 0, 31],
        ]
    )
    pad_ref = torch.BoolTensor(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )
    X, pad = prep_input_and_pad(test_inputs, pad_side="left", device="cpu")

    assert_equal(X_ref, X, msg="prep_input_and_pad produced incorrect X")
    assert_equal(pad_ref, pad, msg="prep_input_and_pad produced incorrect pad")

    # =====================================
    # test prep_pos_from_pad_and_prev_lens
    # =====================================
    # based on the same example as before, compute the associated pos

    pos_ref = torch.LongTensor([[0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]])
    prev_lens = torch.zeros(3, 1).long()
    pos = prep_pos_from_pad_and_prev_lens(pad_ref, prev_lens)

    assert_equal(pos_ref, pos, msg="prep_pos_from_pad_and_prev_lens produced incorrect pos")

    # simulate one step forward of sampling
    seq_lens = (pos[:, -1] + 1).unsqueeze(-1)
    new_pad_ref = torch.BoolTensor(
        [
            [0],
            [0],
            [0],
        ]
    )
    new_pos = prep_pos_from_pad_and_prev_lens(new_pad_ref, seq_lens)

    new_pos_ref = torch.LongTensor([[4], [6], [1]])
    assert_equal(new_pos_ref, new_pos, msg="prep_pos_from_pad_and_prev_lens produced incorrect pos")


@pytest.mark.parametrize("model", REFERENCE_MODELS)
@pytest.mark.parametrize("device", REFERENCE_DEVICES)
def test_sampling_determinism(model, device):
    """
    Some tests for sampling determinism.

    Note: DOES NOT test determinism for nucleus sampling, which is not currently
    compatible with "use_deterministic_algorithms" mode.
    """

    torch.use_deterministic_algorithms(True)

    def reset_seed():
        torch.manual_seed(0)
        np.random.seed(0)

    # ============================================================================
    # Argmax sampling (temperature=0) and categorical sampling with temperature=1
    # ============================================================================

    for temperature in [0, 1]:
        reset_seed()
        xf = get_test_model(model, device)

        # sample 1
        reset_seed()
        s1 = xf.sample("\n", num_tokens=10, temperature=temperature, top_p=None)

        # sample 2
        reset_seed()
        s2 = xf.sample("\n", num_tokens=10, temperature=temperature, top_p=None)

        assert (
            s1["strings"] == s2["strings"]
        ), f"Sampling for {model} was not deterministic with {temperature=}"


@pytest.mark.parametrize("model", REFERENCE_MODELS)
@pytest.mark.parametrize("device", REFERENCE_DEVICES)
def test_batched_sampling_correctness(model, device):
    torch.use_deterministic_algorithms(True)

    def reset_seed():
        torch.manual_seed(0)
        np.random.seed(0)

    reset_seed()
    xf = get_test_model(model, device)
    cfg = xf.cfg
    xf.eval()

    p1 = "if this test works, then these prompts should have the same"
    p2 = "completion regardless of their order"

    # sample 1
    s1 = xf.sample([p1, p2], num_tokens=10, temperature=0, top_p=None)

    # sample 2
    s2 = xf.sample([p2, p1], num_tokens=10, temperature=0, top_p=None)

    # completions for prompt p1
    c1_v1 = s1["strings"][0]
    c1_v2 = s2["strings"][1]

    assert c1_v1 == c1_v2, "Out-of-order prompts resulted in different completions"

    # completions for prompt p2
    c2_v1 = s1["strings"][0]
    c2_v2 = s2["strings"][1]

    assert c2_v1 == c2_v2, "Out-of-order prompts resulted in different completions"
