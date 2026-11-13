# ruff: noqa: E501
"""Export committed results into one JSON file for the research explorer.

Reads only results/<dataset>/<method>/seed0/preds_test.csv, metrics.json and
docs/tables/*.md. Prediction rows carry IDs and scores, never images or text.

    python -m showcase.export_explorer_data
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = ROOT / "docs" / "tables"
OUT = Path(__file__).resolve().parent / "app" / "static" / "data" / "explorer.json"

DATASETS = {
    "fhm": ("FHM", "Hateful memes", "AUROC"),
    "harm_c": ("Harm-C", "Harmful memes · COVID-19", "AUROC"),
    "harm_p": ("Harm-P", "Harmful memes · politics", "AUROC"),
    "mami": ("MAMI", "Misogynous memes", "AUROC"),
    "crisismmd": ("CrisisMMD", "Informative crisis tweets", "AUROC"),
    "mvsa": ("MVSA", "Tweet sentiment · 3 classes", "Accuracy"),
    "food101": ("Food-101", "Recipe pages · 101 classes", "Accuracy"),
}
MAX_POINTS = 3000  # Food-101 test has 22.7k rows; a fixed random subset keeps the page light


def md_table(name: str) -> list[list[str]]:
    rows = []
    for line in (TABLES / name).read_text().splitlines():
        if line.startswith("|") and not re.match(r"^\|[\s\-|]+\|$", line):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def num(s: str) -> float | None:
    s = s.strip().replace("−", "-").replace("<", "")
    try:
        return float(s.split(" ")[0])
    except ValueError:
        return None


def mean_std(cell: str) -> dict | None:
    m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", cell)
    if m:
        return {"mean": float(m[1]), "std": float(m[2])}
    m = re.match(r"([\d.]+)\s*\((\d+)\)", cell)
    if m:
        return {"mean": float(m[1]), "std": None, "seeds": int(m[2])}
    return None


def true_class_prob(row: dict, expert: str, label: int) -> float:
    if f"p_{expert}_true" in row:                       # Food-101 export
        return float(row[f"p_{expert}_true"])
    if f"p_{expert}_{label}" in row:                    # MVSA (3 classes)
        return float(row[f"p_{expert}_{label}"])
    p1 = float(row[f"p_{expert}"])                      # binary: P(y = 1)
    return p1 if label == 1 else 1.0 - p1


def points(dataset: str, method: str) -> dict | None:
    run = RESULTS / dataset / method / "seed0"
    path = run / "preds_test.csv"
    if not path.exists():
        return None
    metrics = json.loads((run / "metrics.json").read_text())
    threshold = metrics.get("val", {}).get("threshold")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) > MAX_POINTS:
        rows = random.Random(0).sample(rows, MAX_POINTS)
    out = []
    for r in rows:
        label = int(r["label"])
        if "pred" in r:
            pred, conf = int(r["pred"]), float(r["prob"])
        else:
            p1 = float(r["prob"])
            pred = int(p1 >= threshold)
            conf = p1 if pred == 1 else 1 - p1
        out.append([
            r["id"], label, pred, round(conf, 4),
            round(float(r["pi_img"]), 4), round(float(r["pi_txt"]), 4), round(float(r["pi_int"]), 4),
            *(round(true_class_prob(r, e, label), 4) for e in ("img", "txt", "int")),
            r.get("quality", "intact"),
        ])
    return {"fields": ["id", "label", "pred", "conf", "pi_img", "pi_txt", "pi_int",
                       "e_img", "e_txt", "e_int", "quality"],
            "rows": out, "n_total": metrics.get("test", {}).get("n") or len(rows),
            "threshold": threshold}


def significance(name: str) -> dict:
    out = {}
    for r in md_table(name)[1:]:
        ci = re.findall(r"[-+−]?\d*\.\d+", r[6])
        out[r[0].replace(" †", "")] = {
            "metric": r[1], "seeds": int(r[2]), "a": num(r[3]), "b": num(r[4]), "delta": num(r[5]),
            "ci": [num(ci[0]), num(ci[1])], "p": r[7], "holm": r[8], "wins": r[9]}
    return out


def main() -> None:
    head, *body = md_table("main.md")
    leaderboard = {"datasets": head[1:],
                   "rows": [{"method": r[0], "scores": [mean_std(c) for c in r[1:]]} for r in body]}
    router_groups = {}
    for r in md_table("mixed_quality_router.md")[1:]:
        router_groups.setdefault(r[0], {})[r[1]] = {
            "n": int(r[2]), "pi": [num(r[3]), num(r[4]), num(r[5])]}
    ablations = [{"variant": r[0], "auroc": num(r[1]), "delta": num(r[2]), "p": r[3]}
                 for r in md_table("ablations.md")[1:]]
    cost_head, *cost_rows = md_table("cost.md")
    cost = [{"method": r[0], **{k: num(v) for k, v in zip(
        ["params_m", "mflops", "ms_b1", "ms_b64", "peak_mb", "train_s"], r[1:], strict=False)}}
        for r in cost_rows]
    transfer = [{"pair": r[0], "scores": dict(zip(md_table("transfer.md")[0][1:],
                                                  (num(c) for c in r[1:]), strict=True))}
                for r in md_table("transfer.md")[1:]]

    from hmf.data.food101 import CLASSES as FOOD
    from hmf.data.mvsa import CLASSES as MVSA
    names = {"fhm": ["benign", "hateful"], "harm_c": ["harmless", "harmful"],
             "harm_p": ["harmless", "harmful"], "mami": ["not misogynous", "misogynous"],
             "crisismmd": ["not informative", "informative"], "mvsa": list(MVSA),
             "food101": [c.replace("_", " ") for c in FOOD]}
    data = {
        "datasets": [{"key": k, "label": v[0], "task": v[1], "metric": v[2], "classes": names[k]}
                     for k, v in DATASETS.items()],
        "clean": {k: points(k, "dynamic") for k in DATASETS},
        "mixed": {k: points(k, "dynamic_mq") for k in DATASETS},
        "router_groups": router_groups,
        "significance": {"clean": significance("significance.md"),
                         "mixed": significance("significance_mq.md")},
        "leaderboard": leaderboard,
        "ablations": ablations,
        "cost": cost,
        "transfer": transfer,
        "source": "results/*/seed0/preds_test.csv and docs/tables/*.md",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    sizes = {k: len(v["rows"]) for k, v in data["clean"].items() if v}
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); clean points {sizes}")


if __name__ == "__main__":
    main()
