import warnings
from dataclasses import dataclass

import blobfile as bf
import pytest
import torch

from neuron_explainer.models import Transformer

SRC_TEST_DATA_FNAME = "https://openaipublic.blob.core.windows.net/neuron-explainer/test-data/test-reference-data/test_data.pt"
DST_TEST_DATA_FNAME = "/tmp/neuron_explainer_reference_test_data.pt"

REFERENCE_MODELS = [
    "gpt2/small",
    "gpt2/medium",
    "gpt2/large",
    "gpt2/xl",
]


@dataclass
class Tolerances:
    max_logits_diff: float
    mean_logits_diff: float
    max_kl: float
    mean_kl: float
    sampling_tolerance: int


DEFAULT_TOLERANCES = Tolerances(
    max_logits_diff=1e-3, mean_logits_diff=5e-5, max_kl=1e-8, mean_kl=1e-9, sampling_tolerance=1
)


def KL(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits1.double(), dim=-1)
    lp1 = torch.log_softmax(logits1.double(), dim=-1)
    lp2 = torch.log_softmax(logits2.double(), dim=-1)
    kl = (p * (lp1 - lp2)).sum(dim=-1)
    return kl


@pytest.mark.parametrize("model_name", REFERENCE_MODELS)
def test_pretrained_models_against_reference_data(model_name: str) -> None:
    """
    Verify that our transformer is correctly loading pretrained models
    by checking their outputs on random data against reference data from huggingface models.
    """

    if not bf.exists(DST_TEST_DATA_FNAME):
        bf.copy(SRC_TEST_DATA_FNAME, DST_TEST_DATA_FNAME)
    else:
        print(f"Test data already exists, reusing.  Run `rm {DST_TEST_DATA_FNAME}` to redownload.")
    with bf.BlobFile(DST_TEST_DATA_FNAME, "rb") as f:
        test_data = torch.load(f)
    data = test_data[model_name]

    print(f"testing model {model_name}")
    xf = Transformer.load(model_name, dtype=torch.float32)
    xf.train(mode=False)
    X = data["inputs"]

    print(f"running forward pass for {model_name}")
    with torch.no_grad():
        Y, _ = xf(X.to(xf.device))
        Y = Y.cpu()

    last_n = data["slice_last_n"]

    logits_slice = Y[:, -last_n:, :]
    target_logits_slice = data["logits_slice"][:, :, :]

    # =========================
    # Get reference tolerance
    # =========================
    tolerance = DEFAULT_TOLERANCES

    # =========================================
    # Look for absolute differences in logits
    # =========================================

    diff = logits_slice - target_logits_slice
    abs_diff = diff.abs()

    print(f"{model_name} | logits diff max: {abs_diff.max()}, diff mean: {abs_diff.mean()}")

    assert (
        abs_diff.max() < tolerance.max_logits_diff
    ), f"max logits diff exceeded {tolerance.max_logits_diff} for {model_name}"
    assert (
        abs_diff.mean() < tolerance.mean_logits_diff
    ), f"mean logits diff exceeded {tolerance.mean_logits_diff} for {model_name}"

    # =========================================
    # Ensure KL is small
    # =========================================

    kl = KL(target_logits_slice, logits_slice)

    print(f"{model_name} | kl max: {kl.max()}, kl mean: {kl.mean()}")

    assert kl.max() < tolerance.max_kl, f"max kl exceeded {tolerance.max_kl} for {model_name}"
    assert kl.mean() < tolerance.mean_kl, f"mean kl exceeded {tolerance.mean_kl} for {model_name}"

    # =========================================
    # Check temperature=0 sampling
    # =========================================

    prompts = [s["prompt"] for s in data["samples"]]

    print(f"sampling for {model_name}")
    results = []
    for idx, p in enumerate(prompts):
        target_tok = data["samples"][idx]["tokens"]
        o = xf.sample(p, temperature=0, num_tokens=len(target_tok))
        results.append(o["tokens"][0] == target_tok)

    assert (
        sum(results) >= len(prompts) - tolerance.sampling_tolerance
    ), f"temperature=0 sampling did not match for {model_name}"

    if sum(results) < len(prompts):
        warnings.warn(f"Not all samples were exact matches for {model_name}")
