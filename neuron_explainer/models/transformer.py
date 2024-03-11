import json
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any, Self, Union

import blobfile as bf
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.checkpoint import checkpoint

from neuron_explainer.models.hooks import (
    AttentionHooks,
    MLPHooks,
    NormalizationHooks,
    TransformerHooks,
)

# for static analysis
Device = Union[torch.device, str]


# NOTE: some code from this file related to attention, MLP, and layernorm operations is copy-pasted in
# neuron_explainer/activations/derived_scalars/reconstituted.py; if those operations change here, they should correspondingly
# be changed in that file.


class SerializableDataclass:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> Self:
        return cls(**d)

    def save(self, path: str) -> None:
        if path.endswith((".pkl", ".pickle")):
            with bf.BlobFile(path, "wb") as f:
                pickle.dump(self.to_dict(), f)
        elif path.endswith(".json"):
            with bf.BlobFile(path, "w") as f:
                json.dump(self.to_dict(), f)
        else:
            raise ValueError(f"Unknown file extension for {path}")

    @classmethod
    def load(cls, path: str) -> Self:
        if path.endswith((".pkl", ".pickle")):
            with bf.BlobFile(path, "rb") as f:
                return cls.from_dict(pickle.load(f))
        elif path.endswith(".json"):
            with bf.BlobFile(path, "r") as f:
                return cls.from_dict(json.load(f))
        else:
            raise ValueError(f"Unknown file extension for {path}")


@dataclass
class TransformerConfig(SerializableDataclass):
    enc: str = "gpt2"
    ctx_window: int = 1024
    d_model: int = 256
    n_layers: int = 2

    # attn
    m_attn: float = 1
    n_heads: int = 8

    # mlp
    m_mlp: float = 4

    @property
    def d_ff(self) -> int:
        return int(self.d_model * self.m_mlp)

    @property
    def d_attn_qk(self) -> int:
        return int(self.d_model * self.m_attn)

    @property
    def d_attn_v(self) -> int:
        return int(self.d_model * self.m_attn)

    @property
    def d_head_qk(self) -> int:
        return safe_div(self.d_attn_qk, self.n_heads)

    @property
    def d_head_v(self) -> int:
        return safe_div(self.d_attn_v, self.n_heads)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_div(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
    return numerator // denominator


# ====================
# Attention utilities
# ====================


@cache
def causal_attn_mask(size: int, device: Device = "cpu") -> Tensor:
    return torch.tril(torch.ones(size, size)).bool().to(device)


def split_heads(Z: Tensor, n_heads: int) -> Tensor:
    batch, seq, d_attn = Z.shape
    return Z.reshape(batch, seq, n_heads, d_attn // n_heads)


def merge_heads(Z: Tensor) -> Tensor:
    batch, seq, n_heads, d_head = Z.shape
    return Z.reshape(batch, seq, n_heads * d_head)


# ===================================
# MLP utilities
# ===================================


def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
    # return x * torch.sigmoid(1.702 * x)


# ========================================
# Sampling, padding and related utilities
# ========================================


def prep_input_and_right_pad_for_forward_pass(
    X: list[list[int]], device: Device = "cpu"
) -> tuple[Tensor, Tensor]:
    # Helper function. The two tensors returned by this function are suitable to be passed to
    # Transformer.forward.
    return prep_input_and_pad(X, "right", device)


def prep_input_and_pad(
    X: list[list[int]], pad_side: str, device: Device = "cpu"
) -> tuple[Tensor, Tensor]:
    # X is a list of tokenized prompts; prompts may have unequal lengths. This function will
    # left-pad X by putting "-1" in all the slots where a prompt is shorter than the longest prompt.
    # Then convert X into a tensor of int tokens. Then build the pad tensor by looking for the
    # "-1"s. Then fill the "-1"s in X with "0"s so the embedding layer doesn't get upset.
    max_len = max([len(prompt) for prompt in X])

    def pad(x):
        padding = [-1] * (max_len - len(x))
        if pad_side == "left":
            return padding + x
        elif pad_side == "right":
            return x + padding
        else:
            raise ValueError(f"pad_side must be 'left' or 'right', not {pad_side}")

    X_tensor = torch.LongTensor([pad(prompt) for prompt in X]).to(device)
    pad = X_tensor == -1
    X_tensor = torch.where(X_tensor == -1, 0, X_tensor)
    return X_tensor, pad


def prep_pos_from_pad_and_prev_lens(pad: Tensor, prev_lens: Tensor) -> Tensor:
    # pad has shape b x s, prev_lens has shape b x 1.
    # For position embedding, we need a tensor of shape (b x s) whose
    # entries are the positions of X in the sequence. When sampling with
    # prompts of unequal length, X is left padded with pad tokens. The
    # position tensor needs to take that into account.
    pos = torch.logical_not(pad).long().cumsum(dim=-1) - 1
    pos = torch.where(pos == -1, 0, pos)
    return pos + prev_lens


def nucleus_sample(logits: Tensor, top_p: float) -> Tensor:
    # top_p in [0,1] is the total probability mass of top outputs.
    # based on https://nn.labml.ai/sampling/nucleus.html
    # input shape: [..., n_vocab] -> output shape: [...]
    sorted_logits, idxs = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # logic to ensure there is always at least one token with nonzero
    # probability when selecting nucleus.
    p0 = cum_probs[..., 0]
    top_p = torch.where(p0 > top_p, p0, top_p)[..., None]

    # sampling
    do_not_sample = cum_probs > top_p
    sorted_logits = sorted_logits.masked_fill(do_not_sample, float("-inf"))
    dist = Categorical(logits=sorted_logits)
    samples = dist.sample()
    tokens = idxs.gather(-1, samples.unsqueeze(-1)).squeeze(-1)
    return tokens


# ===============
# Layer Norm
# ===============


class Norm(nn.Module):
    """LayerNorm reimplementation with hooks."""

    def __init__(
        self,
        size: int,
        eps: float = 1e-5,
        device: Device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.size = size
        self.weight = nn.Parameter(torch.empty(size, **kwargs))  # type: ignore[arg-type]
        self.bias = nn.Parameter(torch.empty(size, **kwargs))  # type: ignore[arg-type]
        self.eps = eps

    def forward(self, x: Tensor, hooks: NormalizationHooks = None) -> Tensor:
        if hooks is None:
            hooks = NormalizationHooks()
        # always do norm in fp32
        orig_dtype = x.dtype
        x = x.float()
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        x = hooks.post_mean_subtraction(x)
        scale = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        scale = hooks.scale(scale)
        x = x / scale
        x = hooks.post_scale(x)
        ret = x * self.weight + self.bias
        return ret.to(orig_dtype)


def apply_layernorm_foldin(ln: Norm, linears: list[nn.Linear]) -> None:
    # folds in a layernorm weight/bias into the next linear layer.
    # ln(x) = W_ln * (x - x.mean())/(x.std()) + b_ln
    # linear(ln(x)) = W_linear * (W_ln * (x - x.mean())/(x.std()) + b_ln) + b_linear

    W_ln = ln.weight.float()
    b_ln = ln.bias.float()
    for linear in linears:
        W_linear = linear.weight.float()
        b_linear = linear.bias.float()

        W_composed = W_linear * W_ln[None, :]

        b_composed = None
        b_composed = b_linear + W_linear @ b_ln

        # should only copy after new weights are calculated
        linear.weight.data.copy_(W_composed)
        linear.bias.data.copy_(b_composed)

    ln.weight.data[:] = 1
    ln.bias.data[:] = 0


# ===========================================
# Attention layers and associated components
# ===========================================


@dataclass
class KeyValueCache:
    """KV cache to save on compute"""

    K_cache: Tensor | None = None  # b x s_old x d
    V_cache: Tensor | None = None  # b x s_old x d
    pad_cache: Tensor | None = None  # b x s_old

    def update(self, K: Tensor, V: Tensor, pad: Tensor):
        # K, V have shape: b x (s_new - s_old) x d
        # pad has shape: b x (s_new - s_old)
        new = self.K_cache is None
        self.K_cache = K if new else torch.cat([self.K_cache, K], dim=1)
        self.V_cache = V if new else torch.cat([self.V_cache, V], dim=1)
        self.pad_cache = pad if new else torch.cat([self.pad_cache, pad], dim=1)
        return self.K_cache, self.V_cache, self.pad_cache


class MultiHeadedDotProductSelfAttention(nn.Module):
    """A configurable multi-headed dot product attention layer."""

    def __init__(
        self,
        cfg: TransformerConfig,
        layer_idx: int,
        device: Device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.n_heads = cfg.n_heads

        # make layers
        kwargs = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_attn_qk, **kwargs)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_attn_qk, **kwargs)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_attn_v, **kwargs)
        self.out_proj = nn.Linear(cfg.d_attn_v, cfg.d_model, **kwargs)
        self.qk_scale = 1 / np.sqrt(np.sqrt(cfg.d_head_qk))

        self.cfg = cfg

    def forward(
        self,
        X: Tensor,
        kv_cache: KeyValueCache | None = None,
        pad: Tensor | None = None,
        hooks: AttentionHooks = AttentionHooks(),
    ) -> tuple[Tensor, KeyValueCache]:
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # update KV cache
        if kv_cache is None:
            kv_cache = KeyValueCache()
        K, V, pad = kv_cache.update(K, V, pad)

        # split apart heads, rescale QK
        Q = split_heads(Q, self.n_heads) * self.qk_scale
        Q = hooks.q(Q)  # bshd
        K = split_heads(K, self.n_heads) * self.qk_scale
        K = hooks.k(K)  # bshd
        V = split_heads(V, self.n_heads)
        V = hooks.v(V)  # bshd

        # useful for calculations below
        n_queries, n_keys = Q.shape[1], K.shape[1]

        # softmax multi-headed dot product attention
        pre_softmax = torch.einsum("bqhd,bkhd -> bhqk", Q, K)

        # apply causal attention mask
        M = causal_attn_mask(n_keys, device=X.device)
        M = M[None, None, -n_queries:]  # make M broadcastable to batch, head
        pre_softmax = pre_softmax.masked_fill(torch.logical_not(M), float("-inf"))

        # apply pad mask
        if pad is not None and torch.any(pad):
            # we only mask out pad tokens for non-pad query tokens
            # (because masking all pad tokens => empty rows => NaNs later)
            pad_mask = torch.bitwise_xor(pad[:, None, :], pad[:, :, None])

            # make pad broadcastable on head dim, and slice for current queries only
            pad_mask = pad_mask[:, None, -n_queries:]

            # apply pad mask
            pre_softmax = pre_softmax.masked_fill(pad_mask, float("-inf"))

        pre_softmax = torch.einsum("bhqk->bqkh", pre_softmax)
        pre_softmax = hooks.qk_logits(pre_softmax)

        pre_softmax = pre_softmax.float()  # for numerical stability
        if hooks.qk_softmax_denominator.is_empty():
            attn = torch.softmax(pre_softmax, dim=-2)
        else:
            # factor out softmax in order to hook
            pre_softmax_max = torch.max(pre_softmax, -2, keepdim=True)[0].detach()
            numerator = torch.exp(pre_softmax - pre_softmax_max)
            denominator = numerator.sum(dim=-2, keepdim=True)
            denominator = hooks.qk_softmax_denominator(denominator)
            attn = numerator / denominator
        attn = attn.to(Q.dtype)

        attn = hooks.qk_probs(attn)

        out = torch.einsum("bqkh,bkhd->bqhd", attn, V)
        out = hooks.v_out(out)
        out = merge_heads(out)  # concatenate results from all heads
        # final output projection
        return self.out_proj(out), kv_cache


# =====================================
# MLP layers and associated components
# =====================================


class MLP(nn.Module):
    """An MLP for a transformer is a simple two-layer network."""

    def __init__(
        self, cfg: TransformerConfig, device: Device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.in_layer = nn.Linear(cfg.d_model, cfg.d_ff, **kwargs)
        self.out_layer = nn.Linear(cfg.d_ff, cfg.d_model, **kwargs)
        self.act = gelu

    def forward(self, X: Tensor, hooks: MLPHooks = MLPHooks()) -> Tensor:
        pre = self.in_layer(X)
        pre = hooks.pre_act(pre)
        a = self.act(pre)
        a = hooks.post_act(a, out_layer=self.out_layer)
        out = self.out_layer(a)
        return out


# =============
# Transformers
# =============


class TransformerLayer(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        layer_idx: int,
        device: Device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.cfg = cfg
        self.attn = MultiHeadedDotProductSelfAttention(cfg, layer_idx, **kwargs)
        self.mlp = MLP(cfg, **kwargs)
        self.ln_1 = Norm(cfg.d_model, **kwargs)
        self.ln_2 = Norm(cfg.d_model, **kwargs)
        self.layer_idx = layer_idx

    def simplify(self) -> None:
        ln_1_linears: list[Any] = [
            self.attn.q_proj,
            self.attn.k_proj,
            self.attn.v_proj,
        ]
        apply_layernorm_foldin(self.ln_1, ln_1_linears)

        ln_2_linears: list[Any] = [self.mlp.in_layer]
        apply_layernorm_foldin(self.ln_2, ln_2_linears)

    def attn_block(
        self, X: Tensor, kv_cache: KeyValueCache | None, pad: Tensor | None, hooks: TransformerHooks
    ) -> Tensor:
        ln_X = self.ln_1(X, hooks.resid.torso.ln_attn)
        ln_X = hooks.resid.torso.post_ln_attn(ln_X)
        attn_delta, kv_cache = self.attn(ln_X, kv_cache, pad, hooks.attn)
        attn_delta = hooks.resid.torso.delta_attn(attn_delta)
        return attn_delta, kv_cache

    def mlp_block(self, X: Tensor, hooks: TransformerHooks) -> Tensor:
        ln_X = self.ln_2(X, hooks.resid.torso.ln_mlp)
        ln_X = hooks.resid.torso.post_ln_mlp(ln_X)
        mlp_delta = self.mlp(ln_X, hooks.mlp)
        mlp_delta = hooks.resid.torso.delta_mlp(mlp_delta)
        return mlp_delta

    def forward(
        self,
        X: Tensor,
        kv_cache: KeyValueCache | None = None,
        pad: Tensor | None = None,
        hooks: TransformerHooks = TransformerHooks(),
    ) -> tuple[Tensor, KeyValueCache]:
        attn_delta, kv_cache = self.attn_block(X, kv_cache, pad, hooks)
        X = X + attn_delta
        X = hooks.resid.torso.post_attn(X)
        mlp_delta = self.mlp_block(X, hooks)
        X = X + mlp_delta
        X = hooks.resid.torso.post_mlp(X)
        return X, kv_cache


class HiddenState:
    """A hidden state for a transformer. Tracks prompt lengths and KV caches."""

    def __init__(self, n_layers: int):
        self.prev_lens = 0
        self.kv_caches = [None for _ in range(n_layers)]

    def set_prev_lens(self, prev_lens) -> None:
        self.prev_lens = prev_lens

    def __getitem__(self, idx: int):
        return self.kv_caches[idx]

    def __setitem__(self, idx: int, value: KeyValueCache | None):
        self.kv_caches[idx] = value


class Transformer(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        # recomputing is optional, and it trades off compute for memory.
        recompute: bool = False,
        device: Device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.enc = tiktoken.get_encoding(self.cfg.enc)
        self.n_vocab = self.enc.n_vocab
        self.recompute = recompute
        self.dtype = dtype

        # build network
        kwargs = {"device": device, "dtype": dtype}
        self.tok_embed = nn.Embedding(self.n_vocab, cfg.d_model, **kwargs)
        self.pos_embed = nn.Embedding(cfg.ctx_window, cfg.d_model, **kwargs)
        self.xf_layers = nn.ModuleList(
            [TransformerLayer(cfg, idx, **kwargs) for idx in range(cfg.n_layers)]
        )
        self.final_ln = Norm(cfg.d_model, **kwargs)
        self.unembed = nn.Linear(cfg.d_model, self.n_vocab, bias=False, **kwargs)

    def simplify(self):
        for xf_layer in self.xf_layers:
            xf_layer.simplify()

        # NOTE: we can't fold layer norm into unembedding layer
        # because it has no bias
        # apply_layernorm_foldin(self.final_ln, [self.unembed])

    @property
    def device(self) -> Device:
        return next(self.parameters()).device

    def set_recompute(self, recompute: bool) -> None:
        self.recompute = recompute

    def forward(
        self,
        tokens: Tensor,
        H: HiddenState | None = None,
        pad: Tensor | None = None,
        hooks: TransformerHooks = TransformerHooks(),
    ) -> tuple[Tensor, HiddenState]:
        """
        Forward pass through the transformer!

        During evaluation or first forward pass in sampling:
            X is expected to be a [batch_size x sequence_length]-shaped LongTensor of encoded prompts.
            H is expected to be None.
            pad is a [batch_size x sequence_length]-shaped boolean Tensor. "1"s mean "ignore this
            token". This parameter must be set if not all encoded prompts in X have the same length.
            Note that activations observed by hooks will include padded values.

        During sampling after first forward pass:
            X is expected to be the new part of the sequences (eg most recently sampled tokens).
            H is expected to have KV-caches of all Keys and Values for prior tokens.
            pad is expected to be None (new tokens are not pad tokens).

        Returns a tuple containing the resulting logits tensor and a new hidden state consisting of a KV cache.
        """
        X, H, pad, hooks = self.run_embed(tokens, H, pad, hooks)
        X, H, pad, hooks = self.run_torso(X, H, pad, hooks)
        return self.run_unembed(X, H, hooks)

    def run_embed(
        self,
        tokens: Tensor,
        H: HiddenState | None = None,
        pad: Tensor | None = None,
        hooks: TransformerHooks = TransformerHooks(),
    ) -> tuple[Tensor, HiddenState, Tensor | None, TransformerHooks]:
        assert tokens.dtype == torch.long, "tokens must be sequences of tokens."
        if H is None:
            H = HiddenState(self.cfg.n_layers)
        if pad is None:
            pad = torch.zeros_like(tokens, dtype=torch.bool)

        # embedding
        X = self.tok_embed(tokens)
        # position encoding logic to support sampling with prompts of unequal length.
        pos = prep_pos_from_pad_and_prev_lens(pad, H.prev_lens)
        seq_lens = (pos[:, -1] + 1).unsqueeze(-1)
        assert all(
            seq_lens <= self.cfg.ctx_window
        ), f"sequences must fit in the context window {self.cfg.ctx_window}."
        H.set_prev_lens(seq_lens)
        X = X + self.pos_embed(pos)

        X = hooks.resid.post_emb(X)
        return X, H, pad, hooks

    def run_torso(
        self,
        X: Tensor,
        H: HiddenState | None,
        pad: Tensor | None,
        hooks: TransformerHooks,
    ) -> tuple[Tensor, HiddenState, Tensor | None, TransformerHooks]:
        # transformer torso
        for i, xf_layer in enumerate(self.xf_layers):
            hooks_layer_i = deepcopy(hooks).bind(layer=i)
            if self.recompute:
                X, H[i] = checkpoint(xf_layer, X, H[i], pad, hooks_layer_i)
            else:
                X, H[i] = xf_layer(X, H[i], pad, hooks_layer_i)
        return X, H, pad, hooks

    def run_ln_f(
        self,
        X: Tensor,
        H: HiddenState | None,
        hooks: TransformerHooks,
    ) -> tuple[Tensor, HiddenState, TransformerHooks]:
        X = self.final_ln(X, hooks.resid.ln_f)
        X = hooks.resid.post_ln_f(X)
        return X, H, hooks

    def run_unembed(
        self,
        X: Tensor,
        H: HiddenState | None,
        hooks: TransformerHooks,
    ) -> tuple[Tensor, HiddenState]:
        # unembedding
        X, H, hooks = self.run_ln_f(X, H, hooks)
        X = self.unembed(X)
        X = hooks.logits(X)
        return X, H

    def sample(
        self,
        prompts: str | list[str] | list[int] | list[list[int]],
        num_tokens: int = 5,
        temperature: float = 1.0,
        top_p: float | None = None,
        hooks: TransformerHooks = TransformerHooks(),
    ) -> dict[str, Any]:
        """
        Sampling with the transformer!

        If top_p is set, then nucleus sampling is used.
        Otherwise, the sampling will be Categorical.
        If temperature=0, sampling is deterministic (and top_p is ignored).

        (Warning: when using torch.use_deterministic_algorithms(True),
        nucleus sampling will throw an error. It depends on torch.cumsum,
        which unfortunately has no deterministic implementation in torch.)

        Output is a dict {'tokens': list[list[int]], 'strings': list[str]}
        """
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if isinstance(prompts[0], str):
            X: list[list[int]] = [self.enc.encode(prompt) for prompt in prompts]
        elif isinstance(prompts[0], int):
            X = [prompts]
        else:
            X = prompts
        X, pad = prep_input_and_pad(X, "left", self.device)
        H = None
        beta = 1 / max(temperature, 1e-10)
        out = {
            "tokens": [[] for _ in prompts],
            "strings": ["" for _ in prompts],
        }

        # sampling loop
        for _ in range(num_tokens):
            with torch.no_grad():
                # get logits
                Y, H = self.forward(X, H, pad, hooks=hooks)
                logits = Y[:, -1] * beta

                # sampling only works if logits are floats
                logits = logits.float()

                # perform sampling
                if temperature == 0:
                    tokens = torch.argmax(logits, dim=-1)
                elif top_p is not None:
                    tokens = nucleus_sample(logits, top_p)
                else:
                    tokens = Categorical(logits=logits).sample()
                X, pad = tokens.unsqueeze(-1), None

            for batch_idx, token in enumerate(tokens):
                out["tokens"][batch_idx].append(token.item())
                out["strings"][batch_idx] += self.enc.decode([token.item()])

        return out

    @classmethod
    def load(
        cls,
        name_or_path: str,
        device: Device | None = None,
        dtype: torch.dtype | None = None,
        simplify: bool = False,
        simplify_kwargs: dict[str, Any] | None = None,
    ) -> "Transformer":
        if bf.exists(name_or_path):
            path = name_or_path
        else:
            path = f"az://openaipublic/neuron-explainer/subject-models/{name_or_path.replace('-', '/')}"
        if not bf.exists(path):
            raise FileNotFoundError(f"Could not find model at {name_or_path}.")
        xf = cls.from_checkpoint(
            path,
            device=device,
            dtype=dtype,
        )
        if simplify:
            if simplify_kwargs is None:
                simplify_kwargs = {}
            xf.simplify(**simplify_kwargs)
        return xf

    def save_checkpoint(
        self,
        path: str,
    ) -> None:
        self.cfg.save(osp.join(path, "config.json"))

        pieces_path = osp.join(path, "model_pieces")
        for k, v in self.state_dict().items():
            with bf.BlobFile(osp.join(pieces_path, f"{k}.pt"), "wb") as f:
                torch.save(v, f)

    def load_state_from_checkpoint(
        self, path: str, device: Device | None = None, dtype: torch.dtype | None = None
    ):
        pieces_path = osp.join(path, "model_pieces")
        piece_files = list(bf.listdir(pieces_path))
        expected_piece_names = set(self.state_dict().keys())
        actual_piece_names = {f[: -len(".pt")] for f in piece_files}
        assert expected_piece_names == actual_piece_names, (
            f"Incorrect pieces at {path}\n\n"
            + f"Missing pieces: {expected_piece_names - actual_piece_names}\n\n"
            + f"Extra pieces: {actual_piece_names - expected_piece_names}"
        )

        if dtype is not None:
            assert isinstance(dtype, torch.dtype), "Must provide valid dtype."
        device = device or self.device

        with ThreadPoolExecutor(max_workers=50) as executor:
            k_to_future = {
                fname[: -len(".pt")]: executor.submit(
                    _load_piece, osp.join(pieces_path, fname), device, dtype
                )
                for fname in piece_files
            }
            d = {k: future.result() for k, future in k_to_future.items()}

        self.load_state_dict(d)

    @classmethod
    def from_checkpoint(
        cls, path: str, device: Device | None = None, dtype: torch.dtype | None = None
    ) -> "Transformer":
        if device is None:
            device = default_device()
        cfg = TransformerConfig.load(osp.join(path, "config.json"))
        xf = cls(cfg, device=device, dtype=dtype)
        xf.load_state_from_checkpoint(path, device=device, dtype=dtype)
        return xf


def _load_piece(
    file_path: str, device: Device, dtype: torch.dtype | None
) -> tuple[str, torch.Tensor]:
    with bf.BlobFile(
        file_path, "rb", cache_dir="/tmp/neuron-explainer-model-pieces-cache", streaming=False
    ) as f:
        t = torch.load(f, map_location=device)
        if dtype is not None:
            t = t.to(dtype)
    return t
