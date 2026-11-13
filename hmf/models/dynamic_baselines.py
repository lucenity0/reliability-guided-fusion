"""Two-expert dynamic-fusion components for issue #23.

Sources checked against both papers and pinned official image/text code:

* QMF: https://proceedings.mlr.press/v202/zhang23ar/zhang23ar.pdf,
  §§4.1–4.2, eqs. 9–18; https://github.com/QingyangZhang/QMF at
  fe6c4c6ef7cb23f0a89594ee413d485f1854268b, under
  text-image-classification/: src/models/late_fusion.py, train_qmf.py
  (rank_loss and model_forward), src/utils/utils.py (History).
* PDF: https://arxiv.org/html/2406.04802v1, §§3.3–4, eqs. 4–10 and
  Appendix C.6; https://github.com/Yinan-Xia/PDF at
  864867426cde076fb3d0529d4df6b440d6963451: src/models/latefusion_pdf.py
  and train_pdf.py.

Shared adaptation: frozen v1 CLIP tokens and our D-017 InputProjection /
UnimodalExpert replace the original trainable image/text encoders. There is
no interaction expert or router. Optimisation and selection use our common
trainer. Binary tasks retain one BCE logit; categorical tasks use C logits.

Planner decision: issue #23 comment 6364114774. Follow official code wherever
it differs from the paper; the following differences are intentional:

* QMF uses T=1, conf=logsumexp(logits)/10, detached in an UNNORMALIZED
  logit sum (no softmax). These values instantiate paper eq. 9. ``pi=None``;
  ``qmf_conf`` holds raw confidences, which may be negative for C>2.
* QMF history accumulates losses from the first batch, with full-training-set
  min/max normalization and rolled in-batch pairs, rather than the sampled
  average in eq. 14. The code's ranking penalty is
  relu(sign(delta_history)*delta_confidence - abs(delta_history)), rather
  than eq. 16. Both modality penalties have coefficient 1.
* The official image/text QMF code covers multiclass tasks. Our binary
  extension computes energy on [-z/2,z/2], preserving sigmoid(z) and energy
  under label exchange; fusion and classification still use the scalar z.
* PDF uses L1 confidence regression, not Appendix C.6 eq. 30's MSE.
  Training uses uncalibrated Co-Belief; eval uses calibrated Co-Belief,
  unlike the paper's description of calibrated fusion throughout. Both are
  softmax-normalized as in code, rather than directly using eq. 10's CCB.
* PDF confidence predictors use three affine maps h->2h->h->1 and a final
  sigmoid, with no intervening activations (the official architecture,
  resized to our expert features). Encoder features are detached during
  confidence training, true-class targets are detached, and fusion weights
  are detached from classification in both modes, as in official code.
* Both objectives SUM the two unimodal CE/BCE losses, plus the fused loss
  and method-specific term; they do not use our dynamic model's lambda_aux.

Numerical/state adaptations: stable logsumexp replaces log(sum(exp)); PDF's
log product is evaluated as a sum of logs, retaining its 1e-8 denominator
offset. Saturated confidences are bounded for logs, uniform DU ties have
calibration 1, and equal-history ranking is zero instead of NaN. QMF history
is checkpointed with ordered train IDs (no validation updates). CPU/device
agnostic tensors replace CUDA-only allocations. Our common train/val
protocol and optimizer replace the original scripts, including selecting
PDF checkpoints with the same calibrated inference used for predictions.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from hmf.models.experts import InputProjection, UnimodalExpert
from hmf.models.registry import register_model

COMMON_KEYS = ("feature_dim", "hidden_dim", "dropout", "num_classes")
QMF_DEFAULTS = {"qmf_temperature": 1.0, "qmf_conf_divisor": 10.0, "qmf_rank_weight": 1.0}
PDF_DEFAULTS = {"pdf_holo_eps": 1e-8, "pdf_confidence_weight": 1.0}


def _official_constants(cfg: dict, defaults: dict) -> None:
    for key, value in defaults.items():
        if cfg[key] != value:
            raise ValueError(f"{key} must remain at the official-code default {value} (#23)")


def energy(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """QMF eq. 13: negative temperature-scaled log partition over classes."""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def collaborative_belief(mono: torch.Tensor, holo_eps: float = 1e-8
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """PDF eqs. 5–6 for two modalities; mono has shape [B, 2].

    Clamp only for logarithms, at machine precision, to handle saturated
    sigmoid outputs. The log-sum form avoids underflow in the paper's product.
    """
    eps = torch.finfo(mono.dtype).eps
    log_mono = mono.clamp(eps, 1 - eps).log()
    holo = log_mono.flip(-1) / (log_mono.sum(dim=-1, keepdim=True) + holo_eps)
    return holo, mono + holo


def relative_calibration(prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PDF eqs. 7–9 on [B, 2, C] class probabilities: return DU and k.

    Equal uniform distributions have k=(1,1); an exactly uniform modality
    paired with a nonuniform modality has k=0. These finite boundary values
    replace the official implementation's 0/0 without changing other ratios.
    """
    du = (prob - 1 / prob.shape[-1]).abs().mean(dim=-1)
    other = du.flip(-1)
    ratio = du / other.clamp_min(torch.finfo(du.dtype).tiny)
    return du, torch.where(du < other, ratio, torch.ones_like(du))


class LossHistory(nn.Module):
    """QMF training-only cumulative losses, bound to ordered training IDs.

    Store CPU float64 history like the official NumPy helper. All training
    rows, including not-yet-visited rows, participate in its min/max range.
    Extra state saves both IDs and losses in best/resume checkpoints; loading
    for inference needs no dataset, while resuming with reordered IDs fails.
    """

    def __init__(self):
        super().__init__()
        self.ids: tuple[str, ...] = ()
        self.values = torch.zeros(0, 2, dtype=torch.float64)

    def prepare(self, ids: Sequence[str]) -> None:
        ids = tuple(ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("QMF history requires nonempty, unique training IDs")
        if self.ids and self.ids != ids:
            raise ValueError("QMF history training IDs differ from the current training role")
        if not self.ids:
            self.ids = ids
            self.values = torch.zeros(len(ids), 2, dtype=torch.float64)

    def get_extra_state(self) -> dict:
        # Python doubles keep history on CPU when torch.load(map_location=device)
        # restores weights; MPS cannot load a float64 tensor onto the device.
        return {"ids": self.ids, "values": self.values.tolist()}

    def set_extra_state(self, state: dict) -> None:
        ids = tuple(state["ids"])
        values = torch.tensor(state["values"], dtype=torch.float64, device="cpu")
        if not ids and values.numel() == 0:
            values = values.reshape(0, 2)
        if self.ids and self.ids != ids:
            raise ValueError("QMF history training IDs differ from the checkpoint")
        if values.shape != (len(ids), 2) or not torch.isfinite(values).all():
            raise ValueError("invalid QMF loss history checkpoint")
        self.ids, self.values = ids, values.clone()

    def _indices(self, indices: torch.Tensor) -> torch.Tensor:
        if not self.ids:
            raise ValueError("call prepare_training(train_ids) before QMF training loss")
        index = indices.detach().cpu()
        if (index.ndim != 1 or index.dtype != torch.int64 or index.numel() == 0
                or index.min() < 0 or index.max() >= len(self.ids)
                or index.unique().numel() != index.numel()):
            raise ValueError("QMF history requires unique in-range training indices")
        return index

    def update(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        index = self._indices(indices)
        values = losses.detach().cpu().double()
        if values.shape != (len(index), 2) or not torch.isfinite(values).all():
            raise ValueError("QMF history requires finite per-sample, per-modality losses")
        self.values[index] += values

    def ranking_loss(self, confidence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Exact pinned-code rolled-pair penalty, summed over the two modalities.

        The code's shifted MarginRankingLoss reduces to
        relu(sign(delta_history) * delta_confidence - abs(delta_history)).
        This differs from the paper's eq. 16. Equal histories and singleton
        batches contribute zero (the official helper divides by zero if every
        history value is equal).
        """
        index = self._indices(indices)
        span = self.values.amax(dim=0) - self.values.amin(dim=0)
        delta = self.values[index] - self.values[index.roll(-1)]
        delta = delta / torch.where(span > 0, span, torch.ones_like(span))
        delta = delta.to(confidence)
        difference = confidence - confidence.roll(-1, dims=0)
        return (delta.sign() * difference - delta.abs()).relu().mean(dim=0).sum()


class _TwoExperts(nn.Module):
    expert_names = ("img", "txt")

    def __init__(self, cfg: dict, required: tuple[str, ...] = ()):
        super().__init__()
        missing = [key for key in COMMON_KEYS + required if key not in cfg]
        if missing:
            raise ValueError(f"config is missing required model keys: {missing}")
        if cfg["num_classes"] < 2:
            raise ValueError("num_classes must be at least 2")
        self.binary = cfg["num_classes"] == 2
        q = 1 if self.binary else cfg["num_classes"]
        d, h, dropout = cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"]
        self.phi_v, self.phi_t = InputProjection(d, h, dropout), InputProjection(d, h, dropout)
        self.img, self.txt = UnimodalExpert(h, dropout, q), UnimodalExpert(h, dropout, q)

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch["text_mask"]
        hi, zi = self.img(self.phi_v(batch["image_tokens"]),
                          self.phi_v(batch["image_pooled"]), None)
        ht, zt = self.txt(self.phi_t(batch["text_tokens"].masked_fill(~mask.unsqueeze(-1), 0)),
                          self.phi_t(batch["text_pooled"]), mask)
        return torch.stack((hi, ht), dim=1), torch.stack((zi, zt), dim=1)

    def classification_losses(self, out: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        z, y = out["z"], batch["label"]
        if self.binary:
            y = y.to(z.dtype)
            main = F.binary_cross_entropy_with_logits(out["logits"].squeeze(-1), y)
            per_expert = F.binary_cross_entropy_with_logits(
                z.squeeze(-1), y[:, None].expand(z.shape[:2]), reduction="none")
        else:
            main = F.cross_entropy(out["logits"], y)
            per_expert = -z.log_softmax(dim=-1).gather(
                -1, y[:, None, None].expand(-1, 2, 1)).squeeze(-1)
        return main, per_expert

    def fuse(self, z: torch.Tensor, weights: torch.Tensor) -> dict:
        # Both official implementations detach fusion weights from classification.
        logits = (weights.detach().unsqueeze(-1) * z).sum(dim=1)
        prob = logits.squeeze(-1).sigmoid() if self.binary else logits.softmax(dim=-1)
        return {"z": z, "logits": logits, "prob": prob}

    def class_probabilities(self, z: torch.Tensor) -> torch.Tensor:
        if self.binary:
            p = z.sigmoid()
            return torch.cat((1 - p, p), dim=-1)
        return z.softmax(dim=-1)

    def loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        parts = self.loss_components(out, batch)
        return sum(parts.values()), {name: value.detach().item() for name, value in parts.items()}


@register_model("qmf")
class QMF(_TwoExperts):
    """Official unnormalized energy fusion and cumulative-loss ranking.

    ``loss`` updates history once before computing the training penalty, as
    in the official loop. ``loss_components`` is read-only; in eval it omits
    ranking, so validation indices never address or modify training history.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg, tuple(QMF_DEFAULTS))
        _official_constants(cfg, QMF_DEFAULTS)
        self.temperature, self.divisor = cfg["qmf_temperature"], cfg["qmf_conf_divisor"]
        self.rank_weight = cfg["qmf_rank_weight"]
        self.history = LossHistory()

    def prepare_training(self, ids: Sequence[str]) -> None:
        self.history.prepare(ids)

    def forward(self, batch: dict) -> dict:
        _, z = self.encode(batch)
        categorical = torch.cat((-z / 2, z / 2), dim=-1) if self.binary else z
        confidence = -energy(categorical, self.temperature) / self.divisor
        return {**self.fuse(z, confidence), "pi": None, "qmf_conf": confidence}

    def loss_components(self, out: dict, batch: dict) -> dict[str, torch.Tensor]:
        main, per_expert = self.classification_losses(out, batch)
        rank = (self.history.ranking_loss(out["qmf_conf"], batch["index"])
                if self.training else out["qmf_conf"].sum() * 0)
        return {"main": main, "aux": per_expert.mean(dim=0).sum(),
                "rank": self.rank_weight * rank}

    def loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        if self.training:
            with torch.no_grad():
                _, per_expert = self.classification_losses(out, batch)
                self.history.update(batch["index"], per_expert)
        return super().loss(out, batch)


@register_model("pdf")
class PDF(_TwoExperts):
    """Official confidence prediction and inference-only relative calibration."""

    def __init__(self, cfg: dict):
        super().__init__(cfg, tuple(PDF_DEFAULTS))
        _official_constants(cfg, PDF_DEFAULTS)
        self.holo_eps, self.confidence_weight = cfg["pdf_holo_eps"], cfg["pdf_confidence_weight"]
        h = cfg["hidden_dim"]
        self.confidence_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(h, 2 * h), nn.Linear(2 * h, h), nn.Linear(h, 1), nn.Sigmoid())
            for _ in self.expert_names)

    def forward(self, batch: dict) -> dict:
        hidden, z = self.encode(batch)
        features = hidden.detach() if self.training else hidden
        mono = torch.cat([head(features[:, k]) for k, head in enumerate(self.confidence_heads)],
                         dim=-1)
        holo, belief = collaborative_belief(mono, self.holo_eps)
        if self.training:
            belief = belief.detach()
            calibration = torch.ones_like(belief)
        else:
            _, calibration = relative_calibration(self.class_probabilities(z))
        pi = (belief * calibration).softmax(dim=-1)
        return {**self.fuse(z, pi), "pi": pi, "mono_conf": mono, "holo_conf": holo,
                "calibration": calibration}

    def loss_components(self, out: dict, batch: dict) -> dict[str, torch.Tensor]:
        main, per_expert = self.classification_losses(out, batch)
        # exp(-CE/BCE) is the true-class probability, also for a negative binary label.
        target = (-per_expert.detach()).exp()
        confidence = (out["mono_conf"] - target).abs().mean(dim=0).sum()
        return {"main": main, "aux": per_expert.mean(dim=0).sum(),
                "confidence": self.confidence_weight * confidence}
