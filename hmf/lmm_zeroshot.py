"""Zero-shot open LMM baselines (design notes §5 row 9; docs/PROTOCOL.md "LMM zero-shot").

Score = P("Yes") from a softmax over the Yes/No next-token logits, so AUROC is
defined. Each class sums the probability mass of its spellings ("Yes", " Yes",
"yes", ...). Greedy decoding only. A response counts as a **refusal or
malformed** when the greedy next token is neither a Yes nor a No spelling; such
samples are scored 0.5 and counted, as PROTOCOL requires.

Usage (on a GPU; the models are 7B and loaded in 4-bit)::

    python -m hmf.lmm_zeroshot --model qwen2.5-vl-7b --dataset fhm --data-root ~/datasets/fhm \\
        --role val --prompt-file configs/lmm_prompt.txt --out-dir results/fhm/lmm_qwen25vl/seed0

``--role test`` additionally needs ``--final`` and this run's config hash in
``results/FINAL_CONFIGS.md`` (PROTOCOL rules 1-2); ``python -m hmf.lmm_zeroshot
--print-hashes`` lists those rows for the frozen prompt. Every run checks that the IDs
it scores equal ``splits/<dataset>/<role>.txt``. Val and test write into one
``metrics.json`` (one block per role), and test reuses the val threshold for
accuracy and macro-F1. LMM baselines apply to the binary datasets only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hmf.config import config_hash
from hmf.data.common import Example
from hmf.eval.guard import check_final, git_sha  # noqa: F401  (re-exported)

MODELS = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-1.6-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
}
# The template is chosen once on FHM val; only this task phrase changes per dataset.
TASK_PHRASES = {
    "fhm": "hateful",
    "harm_c": "harmful",
    "harm_p": "harmful",
    "mami": "misogynistic",
    "crisismmd": "informative about a disaster or crisis",
}
YES_SPELLINGS = ("Yes", " Yes", "yes", " yes", "YES")
NO_SPELLINGS = ("No", " No", "no", " no", "NO")
REFUSAL_SCORE = 0.5


# ── Scoring core (no torch or transformers needed) ──────────────────────────


@dataclass(frozen=True)
class Score:
    prob_yes: float
    refused: bool
    latency_ms: float


def yes_probability(logits: Sequence[float], yes_ids: Iterable[int],
                    no_ids: Iterable[int]) -> tuple[float, bool]:
    """P(Yes) over the two classes, and whether the greedy token is a refusal.

    ``logits`` is the full next-token logit vector. Each class's logit is the
    log-sum-exp over its spellings (its total probability mass), and P(Yes) is
    the two-way softmax of those. Refusal: argmax over the vocabulary is in
    neither id set.
    """
    yes_ids, no_ids = sorted(set(yes_ids)), sorted(set(no_ids))
    if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
        raise ValueError("yes and no token ids must be non-empty and disjoint")

    def lse(ids):
        m = max(logits[i] for i in ids)
        return m + math.log(sum(math.exp(logits[i] - m) for i in ids))

    ly, ln = lse(yes_ids), lse(no_ids)
    prob = 1.0 / (1.0 + math.exp(ln - ly))
    greedy = max(range(len(logits)), key=logits.__getitem__)
    return prob, greedy not in yes_ids and greedy not in no_ids


def load_templates(path: str | Path) -> list[str]:
    """Templates in a prompt file: ``#`` lines are comments, ``---`` separates candidates."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if not line.lstrip().startswith("#")]
    blocks, current = [], []
    for line in lines + ["---"]:
        if line.strip() == "---":
            if "\n".join(current).strip():
                blocks.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    if not blocks:
        raise ValueError(f"{path}: no prompt template found")
    if len(blocks) > 3:
        raise ValueError(f"{path}: at most 3 candidate prompts (PROTOCOL)")
    return blocks


def render_prompt(template: str, example: Example, task: str = "hateful") -> str:
    if "{text}" not in template:
        raise ValueError("prompt template must contain {text}")
    return template.replace("{task}", task).replace("{text}", example.text.strip())


def score_examples(examples: Sequence[Example], template: str,
                   next_token_logits: Callable[[str, str], Sequence[float]],
                   yes_ids: Iterable[int], no_ids: Iterable[int],
                   task: str = "hateful") -> list[Score]:
    """Score each example with ``next_token_logits(image_path, prompt)``."""
    yes_ids, no_ids = list(yes_ids), list(no_ids)
    scores = []
    for example in examples:
        start = time.perf_counter()
        logits = next_token_logits(example.image, render_prompt(template, example, task))
        prob, refused = yes_probability(logits, yes_ids, no_ids)
        latency = (time.perf_counter() - start) * 1000
        scores.append(Score(REFUSAL_SCORE if refused else prob, refused, latency))
    return scores


def run_config(model: str, template: str, dataset: str, quantization: str) -> dict:
    """The fields that identify an LMM baseline run (hashed for FINAL_CONFIGS)."""
    return {
        "method": "lmm_zeroshot",
        "model": MODELS[model],
        "dataset": dataset,
        "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "task_phrase": TASK_PHRASES[dataset],
        "quantization": quantization,
        "decoding": "greedy",
        "refusal_score": REFUSAL_SCORE,
    }


def write_outputs(out_dir: Path, role: str, examples: Sequence[Example], scores: Sequence[Score],
                  meta: dict) -> dict:
    """Write ``preds_<role>.csv`` (id,label,prob) and add the role to ``metrics.json``.

    ``metrics.json`` holds the run fields once at the top level and one block per role,
    so scoring val and then test into the same directory keeps both. Binary metrics use
    the probabilities as written (6 decimals). The threshold for accuracy and macro-F1
    is chosen on val (PROTOCOL rule 3) and reused unchanged on test.
    """
    from hmf.eval.metrics import binary_metrics, find_threshold, threshold_metric

    out_dir.mkdir(parents=True, exist_ok=True)
    split = "test" if role == "test" else "val" if role == "val" else role
    probs = [float(f"{s.prob_yes:.6f}") for s in scores]
    with (out_dir / f"preds_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "label", "prob"])
        for example, prob in zip(examples, probs, strict=True):
            label = "" if example.label is None else example.label
            writer.writerow([example.id, label, f"{prob:.6f}"])
    path = out_dir / "metrics.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if existing.get("config_hash", meta.get("config_hash")) != meta.get("config_hash"):
        raise ValueError(f"{path} belongs to config {existing['config_hash'][:12]}; "
                         "use a separate --out-dir per config")
    latencies = [s.latency_ms for s in scores]
    refused = [e.id for e, s in zip(examples, scores, strict=True) if s.refused]
    block = {
        "role": role,
        "git_sha": meta.get("git_sha"),
        "n": len(scores),
        "n_refused": len(refused),
        "refused_ids": refused,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
    }
    labels = [e.label for e in examples]
    if latencies and None not in labels and len(set(labels)) == 2:
        from hmf.eval.significance import auroc

        block["auroc"] = auroc(labels, probs)
        metric = threshold_metric(meta.get("dataset", ""))
        threshold = (find_threshold(labels, probs, metric)[0] if split == "val"
                     else existing.get("val", {}).get("threshold"))
        if threshold is not None:
            block.update({k: v for k, v in binary_metrics(labels, probs, threshold).items()
                          if k != "auroc"}, threshold_metric=metric)
    path.write_text(json.dumps({**existing, **meta, split: block}, indent=2) + "\n",
                    encoding="utf-8")
    return {**meta, **block}


def check_split_ids(examples: Sequence[Example], dataset: str, role: str,
                    splits_dir: str | Path = "splits") -> None:
    """Refuse to score anything but the committed ``splits/<dataset>/<role>.txt`` IDs."""
    from hmf.data.common import read_id_list

    path = Path(splits_dir) / dataset / f"{role}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; the scored IDs must match a committed list")
    expected, got = set(read_id_list(path)), [e.id for e in examples]
    if len(got) != len(set(got)) or set(got) != expected:
        raise ValueError(f"{dataset} {role}: loaded {len(set(got))} IDs, {path} lists "
                         f"{len(expected)}; {len(set(got) - expected)} extra, "
                         f"{len(expected - set(got))} missing")


QUANTIZATION = "nf4-4bit-fp16"
QWEN_MAX_PIXELS = 1280 * 28 * 28  # 1,003,520 px; recorded in metrics.json (#24)


def final_rows(prompt_file: str | Path = "configs/lmm_prompt.txt") -> list[str]:
    """``results/FINAL_CONFIGS.md`` rows for every (model, dataset) scored with the frozen
    prompt. Printing them reads no data; only the owner adds them to the file."""
    (template,) = load_templates(prompt_file)
    return [f"| {config_hash(run_config(model, template, dataset, QUANTIZATION))} | "
            f"lmm_zeroshot {model} {dataset} ({prompt_file}) | #24 (D-015) | |"
            for model in sorted(MODELS) for dataset in TASK_PHRASES]


# ── Data ────────────────────────────────────────────────────────────────────


def load_role(dataset: str, root: str, role: str, *, allow_test: bool,
              max_samples: int = 0) -> list[Example]:
    if dataset == "fhm":
        from hmf.data.fhm import load_fhm

        return load_fhm(root, role, allow_test=allow_test, max_samples=max_samples)
    if dataset == "crisismmd":
        from hmf.data.crisismmd import load_crisismmd

        annotations, _, images = root.partition(":")
        return load_crisismmd(annotations, images or annotations, role, allow_test=allow_test,
                              max_samples=max_samples)
    if dataset in ("harm_c", "harm_p"):
        from hmf.data.harmeme import load_harmeme

        annotations, _, images = root.partition(":")
        return load_harmeme(annotations, images or annotations, dataset, role,
                            allow_test=allow_test, max_samples=max_samples)
    if dataset == "mami":
        from hmf.data.mami import load_mami

        return load_mami(root, role, allow_test=allow_test, max_samples=max_samples)
    raise ValueError(f"no loader for {dataset!r}")


# ── Hugging Face adapters (GPU only) ───────────────────────────────────────


def load_model(name: str):  # pragma: no cover - needs a GPU and model weights
    """Load a 7B LMM in 4-bit NF4 (fp16 compute, T4-compatible) plus its processor."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    if name == "qwen2.5-vl-7b":
        from transformers import Qwen2_5_VLForConditionalGeneration as Model
    else:
        from transformers import LlavaNextForConditionalGeneration as Model
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = Model.from_pretrained(MODELS[name], quantization_config=quant, device_map="auto")
    model.eval()
    # Qwen2.5-VL reads images at native resolution; a 3000x3000 HarMeme image becomes
    # ~11k visual tokens and runs a T4 out of memory (#24). Cap at 1280*28*28 px
    # (Qwen's documented setting), aspect ratio kept. Every FHM image is below the cap
    # (max 825x823), so the FHM prompt selection is unaffected.
    extra = {"max_pixels": QWEN_MAX_PIXELS} if name == "qwen2.5-vl-7b" else {}
    return model, AutoProcessor.from_pretrained(MODELS[name], **extra)


def token_ids(tokenizer, spellings: Iterable[str]) -> list[int]:
    """First token id of each spelling (single-token spellings only)."""
    ids = []
    for word in spellings:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return sorted(set(ids))


def make_logit_fn(model, processor):  # pragma: no cover - needs a GPU
    import torch
    from PIL import Image

    @torch.inference_mode()
    def next_token_logits(image_path: str, prompt: str) -> list[float]:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        with Image.open(image_path) as image:
            inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
        inputs = inputs.to(model.device)
        return model(**inputs).logits[0, -1].float().tolist()

    return next_token_logits


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU entry point
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--print-hashes"]:  # no GPU, no data: the rows the owner may approve
        print("\n".join(final_rows(*argv[1:2])))
        return 0
    parser = argparse.ArgumentParser(description="Zero-shot LMM baseline: P(Yes) scores. "
                                     "`--print-hashes [PROMPT_FILE]` alone lists the "
                                     "FINAL_CONFIGS rows for the frozen prompt.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset", required=True, choices=sorted(TASK_PHRASES))
    parser.add_argument("--data-root", required=True,
                        help="dataset root (CrisisMMD, Harm-C, Harm-P: '<annotations>:<images>')")
    parser.add_argument("--role", required=True, choices=("val", "test"))
    parser.add_argument("--prompt-file", required=True,
                        help="configs/lmm_prompt.txt, or the candidates file with --candidate")
    parser.add_argument("--candidate", type=int,
                        help="index into a candidates file (prompt selection on FHM val only)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--final-configs", default="results/FINAL_CONFIGS.md")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    templates = load_templates(args.prompt_file)
    if args.candidate is None and len(templates) != 1:
        raise SystemExit("this prompt file has several candidates; pass --candidate")
    if args.candidate is not None and (args.role != "val" or args.dataset != "fhm"):
        raise SystemExit("prompt candidates are compared on FHM val only (PROTOCOL)")
    template = templates[args.candidate or 0]
    cfg = run_config(args.model, template, args.dataset, QUANTIZATION)
    digest = config_hash(cfg)
    if args.role == "test":
        digest = check_final(cfg, args.final, Path(args.final_configs))
    examples = load_role(args.dataset, args.data_root, args.role,
                         allow_test=args.role == "test", max_samples=args.max_samples)
    if not args.max_samples:  # before loading a 7B model: the IDs must be the committed list
        check_split_ids(examples, args.dataset, args.role)
    model, processor = load_model(args.model)
    yes = token_ids(processor.tokenizer, YES_SPELLINGS)
    no = token_ids(processor.tokenizer, NO_SPELLINGS)
    scores = score_examples(examples, template, make_logit_fn(model, processor), yes, no,
                            task=TASK_PHRASES[args.dataset])
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    metrics = write_outputs(Path(args.out_dir), args.role, examples, scores, {
        **cfg, "config_hash": digest, "git_sha": git_sha(), "device": device,
        "gpu_name": device, "torch_version": torch.__version__, "yes_ids": yes, "no_ids": no,
        "max_pixels": QWEN_MAX_PIXELS if args.model == "qwen2.5-vl-7b" else None})
    print(json.dumps({k: metrics.get(k) for k in ("n", "n_refused", "auroc",
                                                  "latency_ms_median")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
