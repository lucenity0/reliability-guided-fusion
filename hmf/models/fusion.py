"""L/P/F expert fusion and detached reliability supervision (MATH_SPEC §4–§6)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from hmf.models.experts import Experts, _mlp
from hmf.models.registry import register_model
from hmf.models.router import Router

REQUIRED_KEYS = (
    "feature_dim", "hidden_dim", "num_heads", "n_cross_layers", "cross_fusion_dim",
    "dropout", "fusion_variant", "router_mode", "tau", "lambda_aux", "lambda_rel",
    "entropy_stop_grad", "num_classes",
)


def reliability_target(expert_losses: torch.Tensor, tau: float) -> torch.Tensor:
    """Detached tempered target; centering preserves ties at very small temperatures."""
    losses = expert_losses.detach()
    return (-(losses - losses.amin(dim=-1, keepdim=True)) / tau).softmax(dim=-1)


@register_model("dynamic")
class Fusion(nn.Module):
    """Shared experts with dynamic, static or global routing.

    Binary tasks use q=1; tasks with num_classes > 2 use q=C. Forward exposes
    expert logits ``z`` [B,K,q], probabilities and routing weights, plus the
    stable log quantities needed by ``loss``. All hyperparameters are required.
    """

    expert_names = ("img", "txt", "int")

    def __init__(self, cfg: dict):
        super().__init__()
        missing = [key for key in REQUIRED_KEYS if key not in cfg]
        if missing:
            raise ValueError(f"config is missing required model keys: {missing}")
        if cfg["fusion_variant"] not in {"L", "P", "F"}:
            raise ValueError("fusion_variant must be L, P or F")
        if cfg["router_mode"] not in {"dynamic", "static", "global"}:
            raise ValueError("router_mode must be dynamic, static or global")
        if cfg["num_classes"] < 2:
            raise ValueError("num_classes must be at least 2")
        for key in ("tau", "lambda_aux", "lambda_rel"):
            value = cfg[key]
            if not math.isfinite(value) or value < 0 or (key != "lambda_rel" and value == 0):
                raise ValueError(f"{key} must be finite and "
                                 f"{'nonnegative' if key == 'lambda_rel' else 'positive'}")
        if not isinstance(cfg["entropy_stop_grad"], bool):
            raise ValueError("entropy_stop_grad must be a boolean")
        # #29 ablations; absent keys keep the full model, so committed hashes are unchanged.
        active = tuple(cfg.get("experts", ",".join(Fusion.expert_names)).split(","))
        if not active or len(set(active)) != len(active) or set(active) - set(Fusion.expert_names):
            raise ValueError(f"experts must be distinct names from {Fusion.expert_names}")
        self.expert_names = active
        router_inputs = tuple(cfg.get("router_inputs", ",".join(Router.INPUTS)).split(","))
        self.variant, self.router_mode = cfg["fusion_variant"], cfg["router_mode"]
        self.tau, self.lambda_aux, self.lambda_rel = (
            cfg["tau"], cfg["lambda_aux"], cfg["lambda_rel"])
        self.binary = cfg["num_classes"] == 2
        q = 1 if self.binary else cfg["num_classes"]
        d, h, dropout = cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"]
        self.experts = Experts(d, h, cfg["num_heads"], cfg["n_cross_layers"],
                               cfg["cross_fusion_dim"], dropout, q)
        self.router = (Router(d, h, len(self.expert_names), dropout, cfg["entropy_stop_grad"],
                              router_inputs)
                       if self.router_mode == "dynamic" else None)
        if self.router_mode == "global":
            self.g = nn.Parameter(torch.zeros(len(self.expert_names)))
        self.fusion_head = _mlp(h, h, q, dropout) if self.variant == "F" else None

    def fuse(self, h: torch.Tensor, z: torch.Tensor, g: torch.Tensor) -> dict:
        """Mix [B,K,h] features / [B,K,q] logits using [B,K] router logits."""
        pi, log_pi = g.softmax(dim=-1), g.log_softmax(dim=-1)
        out = {"pi": pi, "z": z, "log_pi": log_pi}
        if self.variant == "P":
            if self.binary:
                out["prob"] = (pi * z.squeeze(-1).sigmoid()).sum(dim=-1)
            else:
                out["log_prob"] = torch.logsumexp(
                    log_pi.unsqueeze(-1) + z.log_softmax(dim=-1), dim=1)
                out["prob"] = out["log_prob"].exp()
        else:
            logits = ((pi.unsqueeze(-1) * z).sum(dim=1) if self.variant == "L"
                      else self.fusion_head((pi.unsqueeze(-1) * h).sum(dim=1)))
            out["logits"] = logits
            out["prob"] = (logits.squeeze(-1).sigmoid() if self.binary
                           else logits.softmax(dim=-1))
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict:
        experts = self.experts(batch)
        h = torch.stack([experts[name][0] for name in self.expert_names], dim=1)
        z = torch.stack([experts[name][1] for name in self.expert_names], dim=1)
        if self.router_mode == "dynamic":
            g = self.router(batch, z)
        elif self.router_mode == "global":
            g = self.g.expand(z.shape[0], -1)
        else:
            g = z.new_zeros(z.shape[:2])
        return self.fuse(h, z, g)

    def loss_components(self, out: dict, batch: dict) -> dict[str, torch.Tensor]:
        """Differentiable batch means; auxiliary loss also averages over experts."""
        z, y = out["z"], batch["label"]
        if self.binary:
            y = y.to(dtype=z.dtype)
            expert_losses = F.binary_cross_entropy_with_logits(
                z.squeeze(-1), y[:, None].expand(z.shape[:2]), reduction="none")
            main = (F.binary_cross_entropy(out["prob"].clamp(1e-6, 1 - 1e-6), y)
                    if self.variant == "P" else
                    F.binary_cross_entropy_with_logits(out["logits"].squeeze(-1), y))
        else:
            expert_losses = -z.log_softmax(dim=-1).gather(
                -1, y[:, None, None].expand(-1, z.shape[1], 1)).squeeze(-1)
            main = (F.nll_loss(out["log_prob"], y) if self.variant == "P" else
                    F.cross_entropy(out["logits"], y))
        target = reliability_target(expert_losses, self.tau)
        rel = F.kl_div(out["log_pi"], target, reduction="batchmean")
        return {"main": main, "aux": expert_losses.mean(), "rel": rel}

    def loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        parts = self.loss_components(out, batch)
        total = parts["main"] + self.lambda_aux * parts["aux"] + self.lambda_rel * parts["rel"]
        return total, {name: value.detach().item() for name, value in parts.items()}
