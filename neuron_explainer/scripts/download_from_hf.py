import json
import os.path as osp

import click
import torch
from transformers import GPT2LMHeadModel

from neuron_explainer.file_utils import CustomFileHandler
from neuron_explainer.models.transformer import TransformerConfig

EXCLUDES = [".attn.bias", ".attn.masked_bias"]

ALL_MODELS = [
    "gpt2/small",
    "gpt2/medium",
    "gpt2/large",
    "gpt2/xl",
]


def get_hf_model(model_name: str) -> GPT2LMHeadModel:
    _, model_size = model_name.split("/")
    hf_name = "gpt2" if model_size == "small" else f"gpt2-{model_size}"
    model = GPT2LMHeadModel.from_pretrained(hf_name)
    return model


# ====================================
# Conversion from HuggingFace format
# ====================================
def convert(hf_sd: dict) -> dict:
    """convert state_dict from HuggingFace format to our format"""
    n_layers = max([int(k.split(".")[2]) for k in hf_sd.keys() if ".h." in k]) + 1

    hf_to_ours = dict()
    hf_to_ours["wte"] = "tok_embed"
    hf_to_ours["wpe"] = "pos_embed"
    hf_to_ours["ln_f"] = "final_ln"
    hf_to_ours["lm_head"] = "unembed"
    for i in range(n_layers):
        hf_to_ours[f"h.{i}"] = f"xf_layers.{i}"
    hf_to_ours["attn.c_attn"] = "attn.linear_qkv"
    hf_to_ours["attn.c_proj"] = "attn.out_proj"
    hf_to_ours["mlp.c_fc"] = "mlp.in_layer"
    hf_to_ours["mlp.c_proj"] = "mlp.out_layer"

    sd = dict()
    for k, v in hf_sd.items():
        if any(x in k for x in EXCLUDES):
            continue
        if "weight" in k and ("attn" in k or "mlp" in k):
            v = v.T
        k = k.replace("transformer.", "")
        for hf_part, part in hf_to_ours.items():
            k = k.replace(hf_part, part)
        if "attn.linear_qkv." in k:
            qproj, kproj, vproj = v.chunk(3, dim=0)
            sd[k.replace(".linear_qkv.", ".q_proj.")] = qproj
            sd[k.replace(".linear_qkv.", ".k_proj.")] = kproj
            sd[k.replace(".linear_qkv.", ".v_proj.")] = vproj
        else:
            sd[k] = v

    return sd


def download(model_name: str, save_dir: str) -> None:
    assert model_name in ALL_MODELS, f"Must use valid model size, not {model_name=}"
    print(f"Downloading and converting model {model_name} to {save_dir}...")

    print(f"Getting HuggingFace model {model_name}...")
    model = get_hf_model(model_name)

    hf_config = model.config
    base_config = dict(
        enc="gpt2",
        ctx_window=1024,
        # attn
        m_attn=1,
        # mlp
        m_mlp=4,
    )
    cfg = TransformerConfig(
        **base_config,  # type: ignore
        d_model=hf_config.n_embd,
        n_layers=hf_config.n_layer,
        n_heads=hf_config.n_head,
    )

    print("Converting state_dict...")
    sd = convert(model.state_dict())

    print(f"Saving model to {save_dir}...")
    # save to file with config
    pieces_path = osp.join(save_dir, model_name, "model_pieces")
    for k, v in sd.items():
        with CustomFileHandler(osp.join(pieces_path, f"{k}.pt"), "wb") as f:
            torch.save(v, f)

    fname_cfg = osp.join(save_dir, model_name, "config.json")
    with CustomFileHandler(fname_cfg, "w") as f:
        f.write(json.dumps(cfg.__dict__))


@click.command()
@click.argument("save_dir", type=click.Path(exists=False, file_okay=False))
def download_all(save_dir: str) -> None:
    for model_size in ALL_MODELS:
        download(model_size, save_dir)


if __name__ == "__main__":
    download_all()
