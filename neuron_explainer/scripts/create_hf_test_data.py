from typing import Any

import blobfile as bf
import click
import torch
from transformers import GPT2Tokenizer

from neuron_explainer.scripts.download_from_hf import get_hf_model

# ==============================
# Reference models for testing
# ==============================

ALL_MODELS = [
    "gpt2/small",
    "gpt2/medium",
    "gpt2/large",
    "gpt2/xl",
]

# test prompts to sample at temperature zero from
test_prompts = [
    "this is a test",
    "I'm sorry Dave, I'm afraid",
    "We're not strangers to love. You know the rules, and",
    "in the beginning",
    "buy now!",
    "Why did the chicken cross the road?",
]


# =======================================================
# Get the hf models and generate test data from those
# =======================================================


def create_hf_test_data(
    models: list[str],
    test_prompts: list[str],
    num_examples: int,
    seq_len: int,
    sample_len: int,
    last_n: int,
) -> dict:
    # for GPT2 models, seq len maxes out at 1024
    seq_len = min(seq_len, 1024)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompts = [tokenizer.encode(p, return_tensors="pt") for p in test_prompts]

    test_data = {}
    for model_name in models:
        print(f"Creating test data for {model_name}")
        model_data: dict[str, Any] = {}

        # prepare model
        model = get_hf_model(model_name)
        model.cuda()
        print(f"...loaded {model_name}...")

        # make test inputs and get logits
        with torch.no_grad():
            X = torch.randint(0, 50257, (num_examples, seq_len)).cuda()
            Y = model(X)
        X = X.cpu()
        logits = Y.logits.cpu()
        logits_at_inputs = logits.gather(-1, X.unsqueeze(-1)).squeeze(-1)
        logits_slice = logits[:, -last_n:].clone()
        model_data["inputs"] = X
        model_data["logits_at_inputs"] = logits_at_inputs
        model_data["logits_slice"] = logits_slice
        model_data["slice_last_n"] = last_n

        # generate temperature-zero samples
        samples = []
        for op, p in zip(test_prompts, prompts):
            p = p.cuda()
            tok1 = model.generate(p, max_length=sample_len + len(p[0]), temperature=0)
            tok2 = model.generate(p, max_length=sample_len + len(p[0]), temperature=0)

            str1 = tokenizer.decode(tok1[0])
            str2 = tokenizer.decode(tok2[0])
            assert (
                str1 == str2
            ), "HuggingFace temperature-zero generate was unexpectedly nondeterministic"

            # get tokens out as a list, then chop off the ones from the prompt
            tok1 = tok1[0].tolist()
            tok1 = tok1[len(p[0]) :]

            samples.append({"prompt": op, "completion": tokenizer.decode(tok1), "tokens": tok1})

        model_data["samples"] = samples
        test_data[model_name] = model_data

        # free up GPU memory
        model.cpu()
        del model

    return test_data


@click.command()
@click.option(
    "-dir",
    "--savedir",
    type=str,
    default="https://openaipublic.blob.core.windows.net/neuron-explainer/data/test-reference-data",
)
@click.option("-n", "--num_examples", type=int, default=4)
@click.option("-m", "--sample_len", type=int, default=50)
@click.option("-s", "--seq_len", type=int, default=1024)
@click.option("-l", "--last_n", type=int, default=100)
def make_and_save_test_data(
    savedir: str, num_examples: int, seq_len: int, sample_len: int, last_n: int
) -> None:
    test_data = create_hf_test_data(
        models=ALL_MODELS,
        test_prompts=test_prompts,
        num_examples=num_examples,
        seq_len=seq_len,
        sample_len=sample_len,
        last_n=last_n,
    )
    torch.save(test_data, "test_data.pt")
    bf.copy(src="test_data.pt", dst="/".join([savedir, "test_data.pt"]), overwrite=True)


if __name__ == "__main__":
    make_and_save_test_data()
