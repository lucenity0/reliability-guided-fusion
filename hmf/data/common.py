"""Normalised example schema shared by every dataset loader (docs/PROTOCOL.md)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

ROLES = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    """One sample in the shared schema ``{id, image, text, label, split}``.

    ``image`` is a path to the image file (never decoded here), ``label`` is
    0/1 or ``None`` for an unlabeled split, and ``split`` is the dataset's own
    source split name (e.g. ``dev_unseen``), not the protocol role.
    """

    id: str
    image: str
    text: str
    label: int | None
    split: str


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON Lines file, skipping blank lines; errors name the line."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return rows


def check_unique_ids(examples: Iterable[Example], name: str) -> None:
    seen: set[str] = set()
    for example in examples:
        if example.id in seen:
            raise ValueError(f"{name}: duplicate id {example.id!r}")
        seen.add(example.id)


def resolve_role(role: str, roles: Mapping[str, str], *, allow_test: bool) -> str:
    """Map a protocol role to a source split, refusing ``test`` unless allowed.

    Only reporting code (``hmf/evaluate.py --final``) and the integrity checks
    should pass ``allow_test=True`` (docs/PROTOCOL.md, rules 1-2).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if role == "test" and not allow_test:
        raise PermissionError("the test role is for reporting only; pass allow_test=True")
    if set(roles) != set(ROLES):
        raise ValueError(f"roles must map exactly {ROLES}")
    return roles[role]


def write_id_list(examples: Iterable[Example], path: str | Path) -> int:
    """Write one ID per line (IDs only, no content); returns the count."""
    ids = [example.id for example in examples]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return len(ids)


def read_id_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def id_list_record(path: str | Path) -> dict:
    """``{path, n, sha256}`` for an ID list: SHA-256 of its bytes with LF line endings,
    so a CRLF checkout hashes the same as the committed file (#110)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    n = sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
    return {"path": Path(path).as_posix(), "n": n, "sha256": hashlib.sha256(data).hexdigest()}


def stratified_carveout(examples: list[Example], *, fraction: float = 0.10,
                        seed: int = 1234) -> tuple[list[Example], list[Example]]:
    """Split labeled examples into (fit, held_out), stratified by label.

    Each class contributes ``round(fraction * n_class)`` examples to held_out,
    chosen by a seeded permutation. Both parts keep the original file order, so
    the result depends only on the IDs, labels, order, ``fraction`` and ``seed``.
    Used for validation carve-outs (PROTOCOL: FHM Phase 1, MAMI; seed 1234).
    """
    import numpy as np

    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if any(e.label is None for e in examples):
        raise ValueError("carve-outs need labeled examples")
    rng = np.random.default_rng(seed)
    held: set[int] = set()
    for label in sorted({e.label for e in examples}):
        idx = np.array([i for i, e in enumerate(examples) if e.label == label])
        take = round(fraction * idx.size)
        held.update(int(i) for i in rng.permutation(idx)[:take])
    fit = [e for i, e in enumerate(examples) if i not in held]
    out = [e for i, e in enumerate(examples) if i in held]
    return fit, out
