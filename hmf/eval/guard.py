"""Test-split guard shared by every reporting entry point (docs/PROTOCOL.md rules 1-2).

The test role may be read only with ``--final`` and when the run's config hash is
listed in ``results/FINAL_CONFIGS.md``. The owner adds hashes there; code never does.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hmf.config import config_hash

FINAL_CONFIGS = Path("results/FINAL_CONFIGS.md")


def check_final(cfg: dict, final: bool, final_configs: Path = FINAL_CONFIGS) -> str:
    """Return the config hash; refuse the test role unless it is cleared for --final."""
    digest = config_hash(cfg)
    if not final:
        raise PermissionError("the test role needs --final (PROTOCOL rules 1-2)")
    final_configs = Path(final_configs)
    listed = final_configs.read_text(encoding="utf-8") if final_configs.exists() else ""
    if digest not in listed:
        raise PermissionError(f"config hash {digest} is not in {final_configs}; the owner adds "
                              "it before any test-split run")
    return digest


def git_sha() -> str:
    """The commit being run: HMF_GIT_SHA on a Colab VM (the code arrives as an archive,
    not a checkout; scripts/colab/bootstrap.py sets it), otherwise ``git rev-parse``."""
    if os.environ.get("HMF_GIT_SHA"):
        return os.environ["HMF_GIT_SHA"]
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"
