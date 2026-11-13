"""Regenerate every paper table and figure from ``results/`` (#32; design notes §7).

Usage::

    python -m hmf.report                       # results/ -> docs/tables, docs/figures
    python -m hmf.report --results results --out docs --no-figures

Only committed result files are read: ``results/<dataset>/<method>/seed<k>/metrics.json``
and ``preds_test*.csv``. Nothing here selects a model; it summarises frozen runs.

Tables (LaTeX ``.tex`` and Markdown ``.md`` under ``<out>/tables``; numbers also in
``<out>/tables/summary.json``):

- ``main``: test metric per method and dataset, mean ± std over seeds (AUROC; accuracy
  for the multi-class MVSA and Food-101), the seed count where it is below five.
- ``significance``: dynamic vs static (MATH_SPEC §10.1 paired t-test across seeds,
  Holm over the six pre-registered datasets; Food-101 reported outside the family), and
  the same for the D-027 ``_mq`` runs once they exist.
- ``robustness``: D-024, the mean test metric under each corruption for dynamic and
  static, and their paired difference.
- ``calibration``: mean test ECE per method and dataset.
- ``mixed_quality_router``: D-027 router evidence, mean π per expert on image-missing,
  text-missing and intact test samples of the ``dynamic_mq`` runs.
- ``cost``: #31, from ``results/cost/cost.json`` (``hmf.eval.cost``), with LMM latency.
- ``ablations``: #29, FHM validation AUROC per ablation, paired against the full model.
- ``transfer``: #28, from ``results/transfer`` (``hmf.eval.transfer``).

Food-101 runs whose test files were lost take their test metrics from
``results/food101/first_pass_log.json`` (``apply_logged_tests``; D-029).

Figures (PDF and PNG under ``<out>/figures``): π distributions per dataset, validation
curves, and the D-027 router evidence. Figures need matplotlib (``pip install
matplotlib``); ``--no-figures`` skips them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from hmf.eval.significance import holm_bonferroni, paired_seed_ttest

DATASETS = ("fhm", "harm_c", "harm_p", "mami", "crisismmd", "mvsa", "food101")
HOLM_FAMILY = ("fhm", "harm_c", "harm_p", "mami", "crisismmd", "mvsa")  # D-027, #27
DATASET_LABELS = {"fhm": "FHM", "harm_c": "Harm-C", "harm_p": "Harm-P", "mami": "MAMI",
                  "crisismmd": "CrisisMMD", "mvsa": "MVSA", "food101": "Food-101"}
MULTICLASS = {"mvsa", "food101"}
METHODS = ("unimodal_img", "unimodal_txt", "concat_mlp", "linear_probe", "v3_alpha",
           "hateclipper_align", "hateclipper_cross", "lmm_llava16", "lmm_qwen25vl",
           "qmf", "pdf", "static", "global", "dynamic")
METHOD_LABELS = {"unimodal_img": "Image only", "unimodal_txt": "Text only",
                 "concat_mlp": "Concat MLP", "linear_probe": "Linear probe",
                 "v3_alpha": "V3 scalar-$\\alpha$", "hateclipper_align": "Hate-CLIPper Align",
                 "hateclipper_cross": "Hate-CLIPper Cross", "lmm_llava16": "LLaVA-1.6 7B (0-shot)",
                 "lmm_qwen25vl": "Qwen2.5-VL 7B (0-shot)", "qmf": "QMF", "pdf": "PDF",
                 "static": "Static fusion", "global": "Global-$\\pi$",
                 "dynamic": "Dynamic (ours)"}
CONDITIONS = ("drop-image", "drop-text", "noise-image-0.5", "noise-text-0.5",
              "noise-image-1.0", "noise-text-1.0", "swap-image", "swap-text")
GROUPS = ("image_missing", "text_missing", "intact")
EXPERTS = ("img", "txt", "int")


def primary_metric(dataset: str) -> str:
    return "accuracy" if dataset in MULTICLASS else "auroc"


def load_runs(results: Path) -> dict:
    """{(dataset, method): {seed: metrics}} for every run with a metrics.json."""
    runs: dict = defaultdict(dict)
    for path in sorted(results.glob("*/*/seed*/metrics.json")):
        dataset, method, seed_dir = path.parts[-4], path.parts[-3], path.parts[-2]
        if not seed_dir[4:].isdigit():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["_dir"] = str(path.parent)
        runs[(dataset, method)][int(seed_dir[4:])] = metrics
    return dict(runs)


def apply_logged_tests(runs: dict, results: Path) -> int:
    """Fill a missing test block from ``results/<ds>/first_pass_log.json`` (Food-101, D-029).

    The first --final pass printed every run's test metrics but its files were lost; the
    re-score reproduced every run it reached exactly, so the logged values stand in for the
    runs it didn't reach. They carry ``"source": "first_pass_log"``.
    """
    filled = 0
    for path in results.glob("*/first_pass_log.json"):
        logged = json.loads(path.read_text(encoding="utf-8"))
        for tag, values in logged["runs"].items():
            dataset, method, seed_dir = tag.split("/")
            run = runs.get((dataset, method), {}).get(int(seed_dir[4:]))
            if run is not None and not run.get("test"):
                run["test"] = {"accuracy": values["accuracy"],
                               "calibration": {"ece": values["ece"]}, "source": "first_pass_log"}
                filled += 1
    return filled


def test_values(runs: dict, dataset: str, method: str, metric: str | None = None) -> dict:
    """{seed: test metric} for runs that hold a test block with that metric."""
    metric = metric or primary_metric(dataset)
    out = {}
    for seed, m in runs.get((dataset, method), {}).items():
        value = (m.get("test") or {}).get(metric)
        if value is not None:
            out[seed] = float(value)
    return out


def _mean_std(values) -> tuple[float, float, int]:
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return math.nan, math.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def main_table(runs: dict) -> dict:
    table = {}
    for method in METHODS:
        row = {}
        for dataset in DATASETS:
            values = test_values(runs, dataset, method)
            if values:
                mean, std, n = _mean_std(values.values())
                row[dataset] = {"mean": mean, "std": std, "n": n}
        if row:
            table[method] = row
    return table


def compare(runs: dict, a: str, b: str, datasets=DATASETS) -> dict:
    """Paired-by-seed t-test of A - B per dataset; Holm over the HOLM_FAMILY datasets present."""
    rows = {}
    for dataset in datasets:
        va, vb = test_values(runs, dataset, a), test_values(runs, dataset, b)
        seeds = sorted(set(va) & set(vb))
        if len(seeds) < 2:
            continue
        xa, xb = np.array([va[s] for s in seeds]), np.array([vb[s] for s in seeds])
        result = paired_seed_ttest(xa, xb)
        result.update(mean_a=float(xa.mean()), mean_b=float(xb.mean()),
                      wins=int((xa > xb).sum()), seeds=seeds,
                      metric=primary_metric(dataset), in_family=dataset in HOLM_FAMILY)
        rows[dataset] = result
    family = {d: r["p_value"] for d, r in rows.items() if r["in_family"]}
    if family:
        holm = holm_bonferroni(family)
        for dataset in family:
            rows[dataset]["p_holm"] = holm["adjusted"][dataset]
    complete = set(HOLM_FAMILY) <= set(family)
    return {"a": a, "b": b, "rows": rows, "family_complete": complete}


def robustness_table(runs: dict) -> dict:
    out = {}
    for dataset in DATASETS:
        metric = primary_metric(dataset)
        per = {}
        for condition in ("clean", *CONDITIONS):
            values = {}
            for method in ("dynamic", "static"):
                seeds = {}
                for seed, m in runs.get((dataset, method), {}).items():
                    test = m.get("test") or {}
                    block = test if condition == "clean" else (test.get("corruptions") or {}).get(
                        condition)
                    if block and block.get(metric) is not None:
                        seeds[seed] = float(block[metric])
                values[method] = seeds
            common = sorted(set(values["dynamic"]) & set(values["static"]))
            if len(common) < 2:
                continue
            d = np.array([values["dynamic"][s] for s in common])
            s = np.array([values["static"][s] for s in common])
            test = paired_seed_ttest(d, s)
            per[condition] = {"dynamic": float(d.mean()), "static": float(s.mean()),
                              "delta": test["delta"], "p_value": test["p_value"],
                              "n": len(common)}
        if per:
            out[dataset] = per
    return out


def calibration_table(runs: dict) -> dict:
    table = {}
    for method in METHODS:
        row = {}
        for dataset in DATASETS:
            eces = [m["test"]["calibration"]["ece"]
                    for m in runs.get((dataset, method), {}).values()
                    if (m.get("test") or {}).get("calibration")]
            if eces:
                mean, std, n = _mean_std(eces)
                row[dataset] = {"mean": mean, "std": std, "n": n}
        if row:
            table[method] = row
    return table


def read_preds(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mixed_quality_router(runs: dict, method: str = "dynamic_mq") -> dict:
    """Mean π per expert by D-027 group, pooled over the seeds' test predictions."""
    out = {}
    for dataset in DATASETS:
        sums = {g: np.zeros(len(EXPERTS)) for g in GROUPS}
        counts = dict.fromkeys(GROUPS, 0)
        for m in runs.get((dataset, method), {}).values():
            path = Path(m["_dir"]) / "preds_test.csv"
            if not path.exists():
                continue
            for row in read_preds(path):
                group = row.get("quality")
                if group in sums and all(f"pi_{e}" in row for e in EXPERTS):
                    sums[group] += [float(row[f"pi_{e}"]) for e in EXPERTS]
                    counts[group] += 1
        if any(counts.values()):
            out[dataset] = {g: {"n": counts[g], **{e: float(sums[g][i] / counts[g])
                                                   for i, e in enumerate(EXPERTS)}}
                            for g in GROUPS if counts[g]}
    return out


# ── Rendering ────────────────────────────────────────────────────────────────


def _fmt(x: float, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "–"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def _cell(c: dict) -> tuple[str, str]:
    mean, std = _fmt(c["mean"]), _fmt(c["std"])
    few = c["n"] < 5
    md = mean + ("" if c["n"] < 2 else f" ± {std}") + (f" ({c['n']})" if few else "")
    tex = mean + ("" if c["n"] < 2 else r" $\pm$ " + std) + (
        "$^{" + str(c["n"]) + "}$" if few else "")
    return md, tex


def _datasets_in(table: dict) -> list[str]:
    return [d for d in DATASETS if any(d in row for row in table.values())]


def render_grid(table: dict, caption: str, label: str) -> tuple[str, str]:
    datasets = _datasets_in(table)
    head = ["Method", *(DATASET_LABELS[d] for d in datasets)]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{l" + "c" * len(datasets) + "}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"]
    for method, row in table.items():
        cells = [_cell(row[d]) if d in row else ("–", "–") for d in datasets]
        name = METHOD_LABELS.get(method, method)
        md.append("| " + " | ".join([name.replace("$\\alpha$", "α").replace("$\\pi$", "π"),
                                     *(c[0] for c in cells)]) + " |")
        tex.append(" & ".join([name, *(c[1] for c in cells)]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def render_comparison(result: dict, caption: str, label: str) -> tuple[str, str]:
    head = ["Dataset", "Metric", "Seeds", result["a"], result["b"], "Δ", "95% CI", "p",
            "Holm p", "A > B"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{llcccccccc}", r"\toprule",
           " & ".join(h.replace("Δ", r"$\Delta$").replace("_", r"\_").replace("%", r"\%")
                      for h in head) + r" \\", r"\midrule"]
    for dataset, r in result["rows"].items():
        cells = [DATASET_LABELS[dataset] + ("" if r["in_family"] else " †"),
                 r["metric"].upper() if r["metric"] == "auroc" else "Acc.", str(r["n_seeds"]),
                 _fmt(r["mean_a"], 4), _fmt(r["mean_b"], 4), f"{r['delta']:+.4f}",
                 f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]", _fmt_p(r["p_value"]),
                 _fmt_p(r.get("p_holm")), f"{r['wins']}/{r['n_seeds']}"]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(c.replace("<", "$<$") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    note = ("" if result["family_complete"] else
            "\nHolm family incomplete: some of the six pre-registered datasets are missing.\n")
    note += "† outside the Holm family.\n" if any(
        not r["in_family"] for r in result["rows"].values()) else ""
    return "\n".join(md) + "\n" + note, "\n".join(tex) + "\n"


def render_robustness(table: dict) -> tuple[str, str]:
    datasets = [d for d in DATASETS if d in table]
    head = ["Condition", *(DATASET_LABELS[d] for d in datasets)]
    md = ["Dynamic − static under each D-024 condition (mean paired Δ of the test metric; "
          "p from the paired t-test).", "", "| " + " | ".join(head) + " |",
          "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Robustness (D-024): dynamic $-$ static test metric under each "
           r"corruption, mean over paired seeds (p, paired $t$-test).}",
           r"\label{tab:robustness}", r"\begin{tabular}{l" + "c" * len(datasets) + "}",
           r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for condition in ("clean", *CONDITIONS):
        cells = []
        for d in datasets:
            c = table[d].get(condition)
            cells.append("–" if c is None else f"{c['delta']:+.4f} ({_fmt_p(c['p_value'])})")
        md.append("| " + " | ".join([condition, *cells]) + " |")
        tex.append(" & ".join([condition, *(c.replace("<", "$<$") for c in cells)]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def cost_table(results: Path, runs: dict) -> dict | None:
    """#31: rows from ``results/cost/cost.json``, LMM latency and recorded training time."""
    path = results / "cost" / "cost.json"
    if not path.exists():
        return None
    cost = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for method in METHODS:
        model = cost["models"].get(method)
        if model is None:
            continue
        seconds = [m["train_seconds"] for m in runs.get(("mami", method), {}).values()
                   if "train_seconds" in m]
        rows[method] = {"params": model["params"], "flops": model["flops"]["total"],
                        "ms_1": model["latency_ms"]["1"], "ms_64": model["latency_ms"]["64"],
                        "memory_64": model["peak_memory_bytes"].get("64"),
                        "train_seconds": float(np.median(seconds)) if seconds else None}
    lmm = {}
    for method in ("lmm_llava16", "lmm_qwen25vl"):
        values = [m["test"]["latency_ms_mean"] for (d, name), by_seed in runs.items()
                  if name == method for m in by_seed.values()
                  if (m.get("test") or {}).get("latency_ms_mean") is not None]
        if values:
            lmm[method] = float(np.mean(values))
    return {"device": cost.get("gpu_name") or cost["device"], "rows": rows, "lmm_ms": lmm,
            "backbone": cost.get("backbone"), "closed_form": cost.get("closed_form")}


def render_cost(table: dict) -> tuple[str, str]:
    head = ["Method", "Params (M)", "FLOPs (M)", "ms/sample (batch 1)", "ms/sample (batch 64)",
            "Peak MB (batch 64)", "Train s/seed (MAMI)"]
    md = [f"Device: {table['device']}. Fusion heads on cached features; the backbone row is "
          "the frozen CLIP forward.", "", "| " + " | ".join(head) + " |",
          "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{Cost per sample on {table['device']}. FLOPs use two per MAC (A-6). "
           r"Training time is as recorded for each MAMI run, on the GPU that run used.}",
           r"\label{tab:cost}", r"\begin{tabular}{lcccccc}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"]

    def add(name, values):
        md.append("| " + " | ".join([name.replace("$\\alpha$", "α").replace("$\\pi$", "π"),
                                     *values]) + " |")
        tex.append(" & ".join([name, *values]) + r" \\")

    for method, r in table["rows"].items():
        memory = "–" if r["memory_64"] is None else f"{r['memory_64'] / 2**20:.0f}"
        train = "–" if r["train_seconds"] is None else f"{r['train_seconds']:.0f}"
        add(METHOD_LABELS.get(method, method),
            [f"{r['params'] / 1e6:.3f}", f"{r['flops'] / 1e6:.1f}", f"{r['ms_1']:.3f}",
             f"{r['ms_64']:.3f}", memory, train])
    backbone = table["backbone"]
    if backbone:
        flops = backbone["flops"]["image"]["total"] + backbone["flops"]["text"]["total"]
        memory = backbone["peak_memory_bytes"].get("64")
        add("CLIP ViT-L/14@336 (frozen)",
            [f"{backbone['params'] / 1e6:.1f}", f"{flops / 1e6:.0f}",
             f"{backbone['latency_ms']['1']:.2f}", f"{backbone['latency_ms']['64']:.2f}",
             "–" if memory is None else f"{memory / 2**20:.0f}", "–"])
    for method, ms in table["lmm_ms"].items():
        add(METHOD_LABELS[method], ["–", "–", f"{ms:.0f}", "–", "–", "–"])
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


ABLATIONS = ("abl_full", "abl_tau0_25", "abl_tau0_5", "abl_tau1", "abl_tau2", "abl_lrel0",
             "abl_no_img", "abl_no_txt", "abl_no_int", "abl_router_no_entropy",
             "abl_router_no_similarity", "abl_router_no_pooled")
ABLATION_LABELS = {"abl_full": "Full model (tuned, τ = 4)", "abl_tau0_25": "τ = 0.25",
                   "abl_tau0_5": "τ = 0.5", "abl_tau1": "τ = 1", "abl_tau2": "τ = 2",
                   "abl_lrel0": "λ_rel = 0 (no reliability target)",
                   "abl_no_img": "Without image expert", "abl_no_txt": "Without text expert",
                   "abl_no_int": "Without interaction expert",
                   "abl_router_no_entropy": "Router without entropy",
                   "abl_router_no_similarity": "Router without similarity",
                   "abl_router_no_pooled": "Router without pooled embeddings"}


def ablation_table(runs: dict, dataset: str = "fhm") -> dict:
    """#29: val AUROC per variant, paired by seed against the full model (val only)."""
    full = {s: m["val"]["auroc"] for s, m in runs.get((dataset, "abl_full"), {}).items()}
    out = {}
    for name in ABLATIONS:
        values = {s: m["val"]["auroc"] for s, m in runs.get((dataset, name), {}).items()}
        if not values:
            continue
        mean, std, n = _mean_std(values.values())
        row = {"mean": mean, "std": std, "n": n}
        seeds = sorted(set(values) & set(full))
        if name != "abl_full" and len(seeds) >= 2:
            test = paired_seed_ttest(np.array([values[k] for k in seeds]),
                                     np.array([full[k] for k in seeds]))
            row.update(delta=test["delta"], p_value=test["p_value"])
        out[name] = row
    return out


def render_ablation(table: dict) -> tuple[str, str]:
    head = ["Variant", "Val AUROC", "Δ vs full", "p"]
    md = ["FHM validation set, seeds 0–4, every variant trained in the same session; "
          "p from the paired t-test across seeds (uncorrected).", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Ablations on FHM validation (seeds 0--4). $\Delta$ is paired against "
           r"the full model; $p$ from the paired $t$-test, uncorrected.}",
           r"\label{tab:ablations}", r"\begin{tabular}{lccc}", r"\toprule",
           r"Variant & Val AUROC & $\Delta$ & $p$ \\", r"\midrule"]
    for name, r in table.items():
        cells = [f"{r['mean']:.4f} ± {r['std']:.4f}",
                 "–" if "delta" not in r else f"{r['delta']:+.4f}", _fmt_p(r.get("p_value"))]
        md.append("| " + " | ".join([ABLATION_LABELS[name], *cells]) + " |")
        label = (ABLATION_LABELS[name].replace("τ", r"$\tau$").replace("λ_rel", r"$\lambda_{rel}$"))
        tex.append(" & ".join([label, *(c.replace("±", r"$\pm$").replace("<", "$<$")
                                         for c in cells)]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


TRANSFER = ("fhm", "mami", "harm_c", "harm_p")
TRANSFER_METHODS = ("dynamic", "static", "qmf", "pdf", "concat_mlp")


def transfer_table(results: Path, runs: dict) -> dict | None:
    """#28: mean test AUROC per (source, target, method); the diagonal is in-domain --final."""
    root = results / "transfer"
    if not root.exists():
        return None
    cells: dict = defaultdict(list)
    for path in root.glob("*/*/seed*/*.json"):
        r = json.loads(path.read_text(encoding="utf-8"))
        cells[(r["source"], r["target"], r["method"])].append(r["auroc"])
    for dataset in TRANSFER:
        for method in TRANSFER_METHODS:
            values = test_values(runs, dataset, method, "auroc")
            if values:
                cells[(dataset, dataset, method)] = list(values.values())
    return {f"{s}->{t}": {m: {"mean": float(np.mean(v)), "n": len(v)}
                          for (s2, t2, m), v in cells.items() if (s2, t2) == (s, t)}
            for s in TRANSFER for t in TRANSFER}


def render_transfer(table: dict) -> tuple[str, str]:
    head = ["Train → test", *(METHOD_LABELS[m].replace(" (ours)", "") for m in TRANSFER_METHODS)]
    md = ["Mean test AUROC over seeds. Off-diagonal rows score a model on another dataset's "
          "test set with no adaptation; the diagonal (same dataset) is each model's own result. "
          "MAMI-trained dynamic/static models are a 5-seed retrain (the originals were lost).",
          "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Cross-dataset transfer: mean test AUROC, no adaptation.}",
           r"\label{tab:transfer}", r"\begin{tabular}{l" + "c" * len(TRANSFER_METHODS) + "}",
           r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for key, by_method in table.items():
        source, target = key.split("->")
        name = f"{DATASET_LABELS[source]} → {DATASET_LABELS[target]}"
        cells = [f"{by_method[m]['mean']:.3f}" if m in by_method else "–"
                 for m in TRANSFER_METHODS]
        md.append("| " + " | ".join([name, *cells]) + " |")
        tex.append(" & ".join([name.replace("→", r"$\to$"), *cells]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def render_router(table: dict) -> tuple[str, str]:
    head = ["Dataset", "Group", "n", "π img", "π txt", "π int"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Mixed-quality router evidence (D-027): mean $\pi$ per expert on test "
           r"samples by group, dynamic fusion, pooled over seeds.}",
           r"\label{tab:mq-router}", r"\begin{tabular}{llcccc}", r"\toprule",
           r"Dataset & Group & $n$ & $\pi_\mathrm{img}$ & $\pi_\mathrm{txt}$ & "
           r"$\pi_\mathrm{int}$ \\", r"\midrule"]
    for dataset, groups in table.items():
        for group, c in groups.items():
            cells = [DATASET_LABELS[dataset], group.replace("_", " "), str(c["n"]),
                     _fmt(c["img"]), _fmt(c["txt"]), _fmt(c["int"])]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


# ── Figures ──────────────────────────────────────────────────────────────────


def figures(runs: dict, out: Path, router: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
        plt.close(fig)
        written.append(name)

    # π distributions on test, dynamic, pooled over seeds.
    pis = {}
    for dataset in DATASETS:
        values = []
        for m in runs.get((dataset, "dynamic"), {}).values():
            path = Path(m["_dir"]) / "preds_test.csv"
            if path.exists():
                values += [[float(r[f"pi_{e}"]) for e in EXPERTS] for r in read_preds(path)
                           if all(f"pi_{e}" in r for e in EXPERTS)]
        if values:
            pis[dataset] = np.array(values)
    if pis:
        fig, axes = plt.subplots(1, len(pis), figsize=(2.2 * len(pis), 2.6), sharey=True,
                                 squeeze=False)
        for ax, (dataset, values) in zip(axes[0], pis.items(), strict=True):
            ax.boxplot([values[:, i] for i in range(3)], showfliers=False)
            ax.set_xticks([1, 2, 3], ["img", "txt", "int"])
            ax.set_title(DATASET_LABELS[dataset], fontsize=9)
            ax.axhline(1 / 3, color="0.6", lw=0.8, ls="--")
        axes[0][0].set_ylabel(r"router weight $\pi$ (test)")
        save(fig, "pi_distributions")

    # Validation curves (seed 0) for dynamic and static.
    curves = {d: {m: runs.get((d, m), {}).get(0) for m in ("dynamic", "static")}
              for d in DATASETS}
    curves = {d: c for d, c in curves.items() if any(v and v.get("history") for v in c.values())}
    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(2.4 * len(curves), 2.6),
                                 squeeze=False)
        for ax, (dataset, by_method) in zip(axes[0], curves.items(), strict=True):
            metric = primary_metric(dataset)
            for method, m in by_method.items():
                if not m or not m.get("history"):
                    continue
                epochs = [h["epoch"] for h in m["history"]]
                ax.plot(epochs, [h["val"][metric] for h in m["history"]], label=method)
                ax.axvline(m["best_epoch"], color="0.7", lw=0.6)
            ax.set_title(DATASET_LABELS[dataset], fontsize=9)
            ax.set_xlabel("epoch")
        axes[0][0].set_ylabel("val metric (seed 0)")
        axes[0][0].legend(fontsize=7)
        save(fig, "val_curves")

    if router:
        fig, axes = plt.subplots(1, len(router), figsize=(2.4 * len(router), 2.6), sharey=True,
                                 squeeze=False)
        for ax, (dataset, groups) in zip(axes[0], router.items(), strict=True):
            x = np.arange(len(EXPERTS))
            for k, group in enumerate(g for g in GROUPS if g in groups):
                ax.bar(x + (k - 1) * 0.27, [groups[group][e] for e in EXPERTS], width=0.27,
                       label=group.replace("_", " "))
            ax.set_xticks(x, EXPERTS)
            ax.set_title(DATASET_LABELS[dataset], fontsize=9)
        axes[0][0].set_ylabel(r"mean $\pi$ (test)")
        axes[0][0].legend(fontsize=7)
        save(fig, "mq_router_evidence")
    return written


# ── Entry point ──────────────────────────────────────────────────────────────


def build(results: Path, out: Path, *, with_figures: bool = True) -> dict:
    runs = load_runs(results)
    logged = apply_logged_tests(runs, results)
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "main": main_table(runs),
        "significance": compare(runs, "dynamic", "static"),
        "significance_mq": compare(runs, "dynamic_mq", "static_mq"),
        "robustness": robustness_table(runs),
        "calibration": calibration_table(runs),
        "mixed_quality_router": mixed_quality_router(runs),
        "cost": cost_table(results, runs),
        "ablations": ablation_table(runs),
        "transfer": transfer_table(results, runs),
        "logged_test_blocks": logged,
    }
    rendered = {
        "main": render_grid(summary["main"], "Test results: AUROC (accuracy for MVSA and "
                            "Food-101), mean $\\pm$ std over seeds; a superscript gives the "
                            "seed count where it is below five.", "tab:main"),
        "calibration": render_grid(summary["calibration"], "Test expected calibration error "
                                   "(15 bins), mean $\\pm$ std over seeds.", "tab:calibration"),
        "significance": render_comparison(summary["significance"],
                                          "Dynamic vs static fusion on clean test data: paired "
                                          "$t$-test across seeds, Holm over six datasets.",
                                          "tab:significance"),
        "robustness": render_robustness(summary["robustness"]),
    }
    if summary["significance_mq"]["rows"]:
        rendered["significance_mq"] = render_comparison(
            summary["significance_mq"], "Mixed-quality inputs (D-027): dynamic vs static, "
            "paired $t$-test across seeds, Holm over six datasets.", "tab:mq-significance")
    if summary["mixed_quality_router"]:
        rendered["mixed_quality_router"] = render_router(summary["mixed_quality_router"])
    if summary["cost"]:
        rendered["cost"] = render_cost(summary["cost"])
    if summary["ablations"]:
        rendered["ablations"] = render_ablation(summary["ablations"])
    if summary["transfer"]:
        rendered["transfer"] = render_transfer(summary["transfer"])
    for name, (md, tex) in rendered.items():
        (tables_dir / f"{name}.md").write_text(md, encoding="utf-8")
        (tables_dir / f"{name}.tex").write_text(tex, encoding="utf-8")
    (tables_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    summary["tables"] = sorted(rendered)
    summary["figures"] = (figures(runs, out / "figures", summary["mixed_quality_router"])
                          if with_figures else [])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate paper tables and figures.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="docs")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    summary = build(Path(args.results), Path(args.out), with_figures=not args.no_figures)
    print(f"tables: {', '.join(summary['tables'])}")
    print(f"figures: {', '.join(summary['figures']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
