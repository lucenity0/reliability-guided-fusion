"""Best-metric and rolling resume checkpoints.

Ported from V3 ``src/checkpointing.py`` (commit 75c6b9f): ``CheckpointManager``
keeps the best-metric checkpoint and an optional rolling resume checkpoint
written every ``periodic_every`` epochs. Changes from V3: the metric is
generic (val AUROC or accuracy), writes are atomic (a Colab disconnect cannot
leave a truncated file), and the resume state includes torch RNG states.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, best_name: str = "best.pt",
                 periodic_every: int = 0):
        self.dir = Path(checkpoint_dir)
        self.best_path = self.dir / best_name
        self.periodic_path = self.dir / f"{Path(best_name).stem}_last.pt"
        self.periodic_every = periodic_every
        self.best_metric = float("-inf")
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_best_if_improved(self, metric: float, epoch: int, model, extra: dict | None = None
                              ) -> bool:
        """Save the model if ``metric`` strictly beats the best so far."""
        if not metric > self.best_metric:
            return False
        self.best_metric = metric
        _atomic_save({"epoch": epoch, "metric": metric, "model_state": model.state_dict(),
                      **(extra or {})}, self.best_path)
        return True

    def save_periodic(self, epoch: int, model, optimizer, scheduler=None, *,
                      patience_ctr: int = 0, extra: dict | None = None) -> bool:
        if self.periodic_every <= 0 or epoch % self.periodic_every != 0:
            return False
        _atomic_save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": self.best_metric,
            "patience_ctr": patience_ctr,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "extra": extra or {},
        }, self.periodic_path)
        return True

    def load_resume(self, model, optimizer=None, scheduler=None, map_location="cpu"
                    ) -> tuple[int, int, dict]:
        """Restore the rolling checkpoint if present.

        Returns (next epoch to run, patience counter, extra); (1, 0, {}) if none.
        """
        if not self.periodic_path.exists():
            return 1, 0, {}
        ckpt = torch.load(self.periodic_path, map_location=map_location, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        torch.set_rng_state(ckpt["rng_cpu"])
        if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        self.best_metric = ckpt["best_metric"]
        return ckpt["epoch"] + 1, ckpt["patience_ctr"], ckpt["extra"]

    def load_best(self, model, map_location="cpu") -> dict:
        ckpt = torch.load(self.best_path, map_location=map_location, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        return ckpt
