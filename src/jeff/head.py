"""Decision heads over frozen-LM hidden states.

A row with k options uses slots 0..k-1 (slot j = answer label LABELS[j]);
the other slots are masked to -inf. Score rows use their L levels as slots
0..L-1 in ascending order. Optionally the last column (index k_slots) is an
abstain unit, used only by the expected-utility objective and never part of
the option distribution reported to the evaluator. k_slots is 16 by default
(A-P, as in SemIf) and at most 255; rows with more options than the head has
slots raise an error.

Inputs: h = the post-final-norm hidden state at the last prompt position
(what the LM head reads), and optionally `extra` = hidden states of earlier
layers at the same position (see features.py), flattened to [B, n_layers * d].

Kinds (all equal the zero-shot scorer exactly at initialization when
slice-initialized; the tests check this):
- "linear":      z = W h + W_x std(extra) + b. W starts as the LM's unembedding
                 rows of the answer letters ("slice" init), W_x = 0. Weight decay
                 pulls W toward 0, i.e. toward uniform predictions.
- "residual":    z = W0 h + A std([h, extra]) + b, with W0 the frozen letter rows
                 and A, b starting at 0. Weight decay pulls toward the zero-shot
                 model instead of toward uniform predictions.
- "temperature": z = (W0 h) / T(u), T(u) = t_min + softplus(a . u + c), with u =
                 std([h, extra]) plus zero-shot statistics (max probability,
                 normalized entropy, log k). Starts at T = 1. Dividing by a
                 positive number never changes the argmax, so decisions (and
                 accuracy) are always those of the zero-shot model; only the
                 confidence is learned.
std(.) is a per-dimension z-score with mean/std computed on the training set
(set_standardization), stored in the checkpoint.

softcap (models with final-logit soft-capping, e.g. Gemma 4: the LM's logits are
c * tanh(raw / c)): the letter logits W0 h (residual / temperature) or the whole
linear output are capped the same way, so every kind still equals the zero-shot
scorer at initialization. None (the default; all Qwen models) changes nothing.

For residual/temperature the abstain logit is logsumexp(option logits) +
v . u + b_a with b_a = -3 at init (about 5% abstain probability).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .prompting import LABELS, MAX_LABELS, MAX_OPTIONS

KINDS = ("linear", "residual", "temperature", "typed-residual", "typed-regression")


def _inverse_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


class DecisionHead(nn.Module):
    def __init__(self, d_model: int, k_slots: int = MAX_OPTIONS, abstain: bool = True,
                 kind: str = "linear", extra_dim: int = 0, extra_layers: list[int] | None = None,
                 logit_stats: bool = True, t_min: float = 0.05, softcap: float | None = None):
        super().__init__()
        if not 2 <= k_slots <= MAX_LABELS:
            raise ValueError(f"k_slots must be in [2, {MAX_LABELS}]")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.d_model, self.k_slots, self.abstain, self.kind = d_model, k_slots, abstain, kind
        self.softcap = float(softcap) if softcap else None
        self.extra_dim = extra_dim
        self.extra_layers = list(extra_layers or [])
        self.logit_stats = bool(logit_stats) and kind == "temperature"
        self.t_min = t_min
        width = k_slots + int(abstain)
        if kind == "linear":
            self.linear = nn.Linear(d_model, width, bias=True)
            nn.init.zeros_(self.linear.bias)
            nn.init.normal_(self.linear.weight, std=0.02)
            if extra_dim:
                self.extra = nn.Linear(extra_dim, width, bias=False)
                nn.init.zeros_(self.extra.weight)
        else:
            self.register_buffer("base_rows", torch.zeros(k_slots, d_model))
            in_dim = d_model + extra_dim + (3 if self.logit_stats else 0)
            if kind in ("residual", "typed-residual", "typed-regression"):
                self.delta = nn.Linear(in_dim, k_slots * (3 if kind.startswith("typed-") else 1), bias=True)
                nn.init.zeros_(self.delta.weight)
                nn.init.zeros_(self.delta.bias)
                if kind == "typed-regression":
                    self.score_regression = nn.Linear(in_dim, 2, bias=True)
                    nn.init.zeros_(self.score_regression.weight)
                    with torch.no_grad():
                        self.score_regression.bias.copy_(torch.tensor([0.0, _inverse_softplus(0.20)]))
            else:
                self.temp = nn.Linear(in_dim, 1, bias=True)
                nn.init.zeros_(self.temp.weight)
                nn.init.constant_(self.temp.bias, _inverse_softplus(1.0 - t_min))
            if abstain:
                self.abstain_net = nn.Linear(in_dim, 1, bias=True)
                nn.init.zeros_(self.abstain_net.weight)
                nn.init.constant_(self.abstain_net.bias, -3.0)
        if kind != "linear" or extra_dim:
            self.register_buffer("mu", torch.zeros(d_model + extra_dim))
            self.register_buffer("sigma", torch.ones(d_model + extra_dim))

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def slice_init_(self, letter_rows: torch.Tensor) -> None:
        """letter_rows: [n, d_model] with n >= k_slots, row j = unembedding row of
        LABELS[j]; the first k_slots rows are used."""
        if letter_rows.dim() != 2 or letter_rows.shape[0] < self.k_slots or letter_rows.shape[1] != self.d_model:
            raise ValueError(f"letter_rows must be [>= {self.k_slots}, {self.d_model}], got {tuple(letter_rows.shape)}")
        letter_rows = letter_rows[: self.k_slots]
        if self.kind == "linear":
            self.linear.weight[: self.k_slots] = letter_rows.to(self.linear.weight.dtype)
            self.linear.bias[: self.k_slots] = 0.0
        else:
            self.base_rows.copy_(letter_rows.to(self.base_rows.dtype))

    @torch.no_grad()
    def set_standardization(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        if not hasattr(self, "mu"):
            return
        self.mu.copy_(mu.to(self.mu.dtype))
        self.sigma.copy_(sigma.clamp(min=1e-6).to(self.sigma.dtype))

    @property
    def needs_standardization(self) -> bool:
        return hasattr(self, "mu")

    # ---------------------------------------------------------------- forward
    def _std(self, h: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        x = h if not self.extra_dim else torch.cat([h, extra], dim=1)
        return (x - self.mu) / self.sigma

    def _inputs(self, h, k, extra, z0):
        u = self._std(h, extra)
        if self.logit_stats:
            p0 = torch.softmax(z0, dim=1)
            log_k = torch.log(k.float())[:, None]
            entropy = -(p0 * torch.log(p0.clamp(min=1e-12))).sum(dim=1, keepdim=True) / log_k
            u = torch.cat([u, p0.max(dim=1, keepdim=True).values, entropy, log_k], dim=1)
        return u

    def _cap(self, z: torch.Tensor) -> torch.Tensor:
        """Final-logit soft-capping c * tanh(z / c), as in the LM (identity without softcap)."""
        return z if self.softcap is None else self.softcap * torch.tanh(z / self.softcap)

    def _check_extra(self, extra):
        if self.extra_dim and (extra is None or extra.shape[1] != self.extra_dim):
            raise ValueError(f"head expects extra features of width {self.extra_dim}")

    def check_fits(self, k) -> None:
        """Raise if a row has more options than the head has slots."""
        widest = (int(k.max()) if k.numel() else 0) if isinstance(k, torch.Tensor) else max(k, default=0)
        if widest > self.k_slots:
            raise ValueError(f"a row has {widest} options but the head has {self.k_slots} slots; "
                             f"use a head with head.k_slots >= {widest}")

    def option_logits(self, h: torch.Tensor, k: torch.Tensor, extra: torch.Tensor | None = None, q=None):
        """-> (masked option logits [B, K], abstain logit [B, 1] or None)."""
        self._check_extra(extra)
        self.check_fits(k)
        invalid = torch.arange(self.k_slots, device=h.device)[None, :] >= k[:, None]
        if self.kind == "linear":
            z = self.linear(h)
            if self.extra_dim:
                extra_std = (extra - self.mu[self.d_model:]) / self.sigma[self.d_model:]
                z = z + self.extra(extra_std)
            z = self._cap(z)
            options = z[:, : self.k_slots].masked_fill(invalid, float("-inf"))
            return options, (z[:, self.k_slots:] if self.abstain else None)
        raw = self._cap(h @ self.base_rows.T)
        z0 = raw.masked_fill(invalid, float("-inf"))
        u = self._inputs(h, k, extra, z0)
        if self.kind in ("residual", "typed-residual", "typed-regression"):
            delta = self.delta(u)
            if self.kind.startswith("typed-"):
                if q is None:
                    raise ValueError("typed heads require explicit question types")
                delta = delta.reshape(len(h), 3, self.k_slots)[torch.arange(len(h), device=h.device), q]
            options = z0 + delta
            if self.kind == "typed-regression":
                # Continuous location/scale supplies the required ordinal distribution.
                reg = self.score_regression(u)
                center, spread = reg[:, :1].sigmoid(), F.softplus(reg[:, 1:]) + 0.02
                levels = torch.arange(self.k_slots, device=h.device)[None, :] / (k[:, None]-1)
                score_logits = (-0.5 * ((levels-center)/spread).square()).masked_fill(invalid,float("-inf"))
                options = torch.where((q==2)[:,None],score_logits,options)
        else:  # scale before masking: -inf / T would give NaN gradients for T
            options = (raw / (self.t_min + F.softplus(self.temp(u)))).masked_fill(invalid, float("-inf"))
        abstain = (torch.logsumexp(options, dim=1, keepdim=True) + self.abstain_net(u)) if self.abstain else None
        return options, abstain

    def masked_logits(self, h: torch.Tensor, k: torch.Tensor, extra: torch.Tensor | None = None,
                      use_abstain: bool = False, q=None) -> torch.Tensor:
        """[B, K(+1)] logits; invalid slots at -inf; the abstain column (last)
        is -inf unless use_abstain=True."""
        options, abstain = self.option_logits(h, k, extra, q=q)
        if not self.abstain:
            return options
        if not use_abstain:
            abstain = torch.full_like(abstain, float("-inf"))
        return torch.cat([options, abstain], dim=1)

    forward = masked_logits

    def temperature_of(self, h, k, extra=None):
        """Per-row temperature T(u) (temperature kind only), for analysis."""
        if self.kind != "temperature":
            raise ValueError("only the temperature head has a per-input temperature")
        self._check_extra(extra)
        self.check_fits(k)
        invalid = torch.arange(self.k_slots, device=h.device)[None, :] >= k[:, None]
        z0 = self._cap(h @ self.base_rows.T).masked_fill(invalid, float("-inf"))
        return (self.t_min + F.softplus(self.temp(self._inputs(h, k, extra, z0)))).squeeze(1)

    # --------------------------------------------------------------- storage
    def config(self) -> dict:
        config = {"d_model": self.d_model, "k_slots": self.k_slots, "abstain": self.abstain,
                  "kind": self.kind, "extra_dim": self.extra_dim, "extra_layers": self.extra_layers,
                  "logit_stats": self.logit_stats, "t_min": self.t_min}
        if self.softcap is not None:  # soft-capped models only: other checkpoints stay as before
            config["softcap"] = self.softcap
        return config

    def save(self, path, metadata: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config(), "metadata": metadata}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> tuple["DecisionHead", dict]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        head = cls(**payload["config"])  # old checkpoints: linear kind, no extras
        head.load_state_dict(payload["state_dict"])
        return head, payload.get("metadata", {})


class SlotHead(DecisionHead):
    """Backward-compatible name for the linear head over the last layer only."""

    def __init__(self, d_model: int, k_slots: int = MAX_OPTIONS, abstain: bool = True):
        super().__init__(d_model, k_slots, abstain, kind="linear")

    @classmethod
    def load(cls, path, map_location="cpu"):
        return DecisionHead.load(path, map_location)


def collect_letter_rows(model, tokenizer, k_slots: int = MAX_OPTIONS) -> torch.Tensor:
    """[k_slots, d_model] fp32 rows of the LM output embedding at the ids of the
    first k_slots answer labels (precompute saves all MAX_LABELS).

    Uses get_output_embeddings(), which resolves tied embeddings correctly.
    Saved next to feature caches so head training never loads the base model.
    """
    from .prompting import slot_ids

    ids = slot_ids(tokenizer, k_slots)
    weight = model.get_output_embeddings().weight.detach()
    return weight[ids].float().cpu().clone()


def save_letter_rows(path, rows: torch.Tensor, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"letter_rows": rows, "letters": list(LABELS[: rows.shape[0]]), "metadata": metadata}, path)


def load_letter_rows(path, k_slots: int | None = None) -> torch.Tensor:
    """[n, d] label rows as saved (n = 16 in older caches, 255 now), or the first k_slots."""
    rows = torch.load(Path(path), map_location="cpu", weights_only=True)["letter_rows"]
    if k_slots is None:
        return rows
    if rows.shape[0] < k_slots:
        raise ValueError(f"{path} has rows for {rows.shape[0]} answer labels but the head has {k_slots} slots; "
                         f"rerun scripts/precompute.py, which saves all {MAX_LABELS}")
    return rows[:k_slots]


def model_softcap(model) -> float | None:
    """The LM's final-logit soft-capping constant c (Gemma 4: 30.0), or None (Qwen)."""
    config = model.config
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    value = getattr(config, "final_logit_softcapping", None)
    return float(value) if value else None


def letter_rows_softcap(path) -> float | None:
    """The soft-capping constant saved with letter_rows.pt by precompute (None for uncapped models)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return (payload.get("metadata") or {}).get("final_logit_softcapping")


def load_head_safetensors(folder, map_location="cpu") -> DecisionHead:
    """Release format: head.safetensors (state dict) + head_config.json (constructor arguments); no pickle."""
    import json
    from safetensors.torch import load_file
    folder = Path(folder)
    head = DecisionHead(**json.loads((folder / "head_config.json").read_text()))
    head.load_state_dict(load_file(str(folder / "head.safetensors"), device=str(map_location)))
    return head
