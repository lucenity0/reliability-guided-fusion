"""Frozen CLIP feature extraction into feature caches (docs/MATH_SPEC.md §1, issue #11).

Cache versions (``SPECS``):

* ``v1``: ``openai/clip-vit-large-patch14-336`` at 336 px. Every token of the last
  vision hidden state goes through ``post_layernorm`` and ``visual_projection``; the
  24×24 patch grid is then 2×2 average-pooled, giving [CLS; 144 patches] = 145 tokens.
  Every text token goes through ``text_projection`` (the text ``last_hidden_state``
  already has ``final_layer_norm`` applied). The CLS row and the EOS token are CLIP's
  ``get_image_features`` / ``get_text_features``; ``similarity`` is their cosine.
* ``v1_full`` (#30): as ``v1`` but without pooling the patch grid: [CLS; 576 patches] =
  577 tokens, the full-resolution token cache.
* ``siglip`` (#30): ``google/siglip-so400m-patch14-384`` at 384 px, width 1152. SigLIP has no
  CLS token: row 0 is the vision head's attention-pooled output (``get_image_features``),
  followed by the 27×27 post-LayerNorm patch grid 3×3 average-pooled to 9×9 (82 tokens).
  Text uses SigLIP's 64-token context padded to full length, as it was trained; tokens are
  the final hidden states and ``text_pooled`` is ``get_text_features``. Hidden states are
  not projected, because SigLIP's towers share their width.
* ``hc224``: ``openai/clip-vit-large-patch14`` at 224 px, Hate-CLIPper's inputs (#13):
  the pre-projection pooler outputs, ``image_tokens`` [N, 1, 1024] (post-LayerNorm CLS)
  and ``text_tokens`` [N, 1, 768] (EOS after the final LayerNorm). ``similarity`` uses
  that checkpoint's projected embeddings. Images are first squashed to 224×224 with
  Pillow's default filter, as Hate-CLIPper's FHM loader does (datasets.py@86ad7f1 L47),
  so CLIP's centre crop keeps the whole meme; the manifest records the filter and Pillow.

One cache per source split, at ``<out>/<dataset>/<cache_version>/<split>/`` (format:
``hmf/features/cache.py``). Roles map to source splits through the loader's role map,
so FHM's train and val roles share the ``train`` cache and pick their rows with
``splits/fhm/{train,val}.txt``. The test role is cached only with ``--allow-test``
(D-021), and its manifest records ``allow_test: true``; caching computes no metric
and selects nothing.

CLIP is frozen and runs in fp32 under ``torch.inference_mode()`` (D-020;
``--compute-dtype``); arrays are always stored in fp16. Text is truncated to CLIP's
77-token context. A CUDA out-of-memory error halves the batch size for the
rest of the split, and the size used is recorded in the manifest.

    python -m hmf.features.extract --dataset fhm --root ~/datasets/fhm \\
        --roles train val test --allow-test
    python -m hmf.features.extract --dataset harm_c --root ANNOTATIONS:IMAGES ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hmf.data.common import ROLES, Example, resolve_role
from hmf.eval.guard import git_sha
from hmf.features.cache import UNLABELED, load_cache, write_cache

MAX_TEXT_TOKENS = 77
PATCH_POOL = 2
MAX_REL, MIN_COS = 1e-3, 0.9999  # D-020 acceptance against fp32 CLIP
OUTPUT_KEYS = ("image_tokens", "text_tokens", "text_mask", "text_pooled", "similarity")


@dataclass(frozen=True)
class CacheSpec:
    backbone: str
    image_size: int
    pooled_only: bool  # pre-projection pooler outputs only (Hate-CLIPper, #13)
    # Squash to this size (Pillow's default filter) before the CLIP processor, as
    # Hate-CLIPper's FHM loader does (datasets.py@86ad7f1 L47); None keeps CLIP's own
    # aspect-preserving resize and centre crop.
    pre_resize: tuple[int, int] | None = None
    family: str = "clip"  # "clip" or "siglip" (#30)
    patch_pool: int = PATCH_POOL  # 1 keeps every patch token
    max_text_tokens: int = MAX_TEXT_TOKENS


SPECS = {
    "v1": CacheSpec("openai/clip-vit-large-patch14-336", 336, pooled_only=False),
    "v1_full": CacheSpec("openai/clip-vit-large-patch14-336", 336, pooled_only=False,
                         patch_pool=1),
    "siglip": CacheSpec("google/siglip-so400m-patch14-384", 384, pooled_only=False,
                        family="siglip", patch_pool=3, max_text_tokens=64),
    "hc224": CacheSpec("openai/clip-vit-large-patch14", 224, pooled_only=True,
                       pre_resize=(224, 224)),
}

# One batch from the data pipeline: (rows, pixel_values, input_ids, attention_mask,
# number of texts truncated to MAX_TEXT_TOKENS).
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
Encode = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]


# ── Datasets ───────────────────────────────────────────────────────────────


DATASETS = ("crisismmd", "fhm", "food101", "harm_c", "harm_p", "mami", "mvsa")


def role_map(dataset: str) -> Mapping[str, str]:
    if dataset == "fhm":
        from hmf.data.fhm import DEFAULT_ROLES
    elif dataset == "mami":
        from hmf.data.mami import DEFAULT_ROLES
    elif dataset == "crisismmd":
        from hmf.data.crisismmd import DEFAULT_ROLES
    elif dataset in ("harm_c", "harm_p"):
        from hmf.data.harmeme import DEFAULT_ROLES
    elif dataset == "mvsa":
        from hmf.data.mvsa import DEFAULT_ROLES
    elif dataset == "food101":
        from hmf.data.food101 import DEFAULT_ROLES
    else:
        raise ValueError(f"no loader for {dataset!r} yet")
    return DEFAULT_ROLES


def load_source(dataset: str, root: str, split: str) -> list[Example]:
    """All examples of one source split, in file order.

    FHM and MAMI take their release directory. The others take ``ANNOTATIONS:IMAGES``
    (images default to the annotations directory), as ``hmf.lmm_zeroshot.load_role`` does.
    """
    if dataset == "fhm":
        from hmf.data.fhm import load_fhm_split

        return load_fhm_split(root, split)
    if dataset == "mami":
        from hmf.data.mami import load_mami_split

        return load_mami_split(root, split)
    annotations, _, images = root.partition(":")
    images = images or annotations
    if dataset == "crisismmd":
        from hmf.data.crisismmd import load_crisismmd_split

        return load_crisismmd_split(annotations, images, split)
    if dataset in ("harm_c", "harm_p"):
        from hmf.data.harmeme import load_harmeme_split

        return load_harmeme_split(annotations, images, dataset, split)
    if dataset == "mvsa":
        from hmf.data.mvsa import load_mvsa_split

        return load_mvsa_split(annotations, images, split)
    if dataset == "food101":
        from hmf.data.food101 import load_food101_split

        return load_food101_split(annotations, images, split)
    raise ValueError(f"no loader for {dataset!r} yet")


def source_splits(dataset: str, roles: Sequence[str], *, allow_test: bool = False) -> list[str]:
    """Source splits behind the roles, deduplicated; carve-out parts map to their source."""
    splits = []
    for role in roles:
        split = resolve_role(role, role_map(dataset), allow_test=allow_test).split(":")[0]
        if split not in splits:
            splits.append(split)
    return splits


# ── CLIP ───────────────────────────────────────────────────────────────────


def pool_patch_grid(tokens: torch.Tensor, pool: int = PATCH_POOL) -> torch.Tensor:
    """Keep the CLS row and average-pool the square patch grid.

    [B, 1 + g², d] -> [B, 1 + (g / pool)², d]. Patches are row-major, as CLIP's patch
    embedding flattens them, and so is the output.
    """
    cls, patches = tokens[:, :1], tokens[:, 1:]
    batch, count, dim = patches.shape
    side = math.isqrt(count)
    if side * side != count or side % pool:
        raise ValueError(f"{count} patches do not form a square grid divisible by {pool}")
    grid = patches.reshape(batch, side, side, dim).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, pool).flatten(2).transpose(1, 2)
    return torch.cat([cls, pooled], dim=1)


def pooler(output) -> torch.Tensor:
    """``get_*_features`` return a tensor before transformers 5 and an output object after."""
    return output if torch.is_tensor(output) else output.pooler_output


def encode_batch(model, spec: CacheSpec, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cache rows for one batch from a ``transformers`` ``CLIPModel`` (or ``SiglipModel``)."""
    if spec.family == "siglip":
        return _encode_siglip(model, spec, pixel_values, input_ids, attention_mask)
    vision = model.vision_model(pixel_values=pixel_values)
    text = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    image_embed = model.visual_projection(vision.pooler_output)
    text_embed = model.text_projection(text.pooler_output)
    if spec.pooled_only:
        image_tokens = vision.pooler_output[:, None]
        text_pooled = text.pooler_output
        text_tokens = text_pooled[:, None]
        text_mask = torch.ones(text_pooled.shape[0], 1, dtype=torch.bool)
    else:
        hidden = model.vision_model.post_layernorm(vision.last_hidden_state)
        image_tokens = pool_patch_grid(model.visual_projection(hidden), spec.patch_pool)
        text_tokens = model.text_projection(text.last_hidden_state)
        text_pooled = text_embed
        text_mask = attention_mask.bool()
    similarity = F.cosine_similarity(image_embed.float(), text_embed.float(), dim=-1)
    return {"image_tokens": image_tokens, "text_tokens": text_tokens, "text_mask": text_mask,
            "text_pooled": text_pooled, "similarity": similarity}


def _encode_siglip(model, spec: CacheSpec, pixel_values, input_ids, attention_mask) -> dict:
    vision = model.vision_model(pixel_values=pixel_values)
    text = model.text_model(input_ids=input_ids)  # SigLIP attends over its padding, as trained
    image_embed, text_embed = vision.pooler_output, text.pooler_output
    image_tokens = pool_patch_grid(torch.cat([image_embed[:, None], vision.last_hidden_state],
                                             dim=1), spec.patch_pool)
    similarity = F.cosine_similarity(image_embed.float(), text_embed.float(), dim=-1)
    return {"image_tokens": image_tokens, "text_tokens": text.last_hidden_state,
            "text_mask": attention_mask.bool(), "text_pooled": text_embed,
            "similarity": similarity}


@dataclass
class Preprocess:
    """Collate function: CLIP image processing and tokenisation (runs in loader workers)."""

    image_processor: object
    tokenizer: object
    pre_resize: tuple[int, int] | None = None
    max_text_tokens: int = MAX_TEXT_TOKENS

    def __call__(self, batch: list[tuple[int, object, str]]) -> Batch:
        rows, images, texts = zip(*batch, strict=True)
        if self.pre_resize is not None:  # hc224: Hate-CLIPper's datasets.py L47
            images = [image.convert("RGB").resize(self.pre_resize) for image in images]
        pixels = self.image_processor(images=list(images), return_tensors="pt")["pixel_values"]
        tokens = self.tokenizer(list(texts), padding="max_length", truncation=True,
                                max_length=self.max_text_tokens, return_tensors="pt")
        full = self.tokenizer(list(texts), verbose=False)["input_ids"]
        truncated = sum(len(ids) > self.max_text_tokens for ids in full)
        mask = tokens.get("attention_mask")
        if mask is None:  # SigLIP's tokenizer returns no mask: mark the real tokens
            lengths = torch.tensor([min(len(ids), self.max_text_tokens) for ids in full])
            mask = (torch.arange(self.max_text_tokens)[None] < lengths[:, None]).long()
        return (torch.tensor(rows), pixels, tokens["input_ids"], mask, truncated)


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, row: int) -> tuple[int, object, str]:
        from PIL import Image

        example = self.examples[row]
        try:
            with Image.open(example.image) as image:
                return row, image.convert("RGB"), example.text
        except OSError as exc:
            raise OSError(f"cannot read the image of {example.id!r}: {example.image}") from exc


def make_batches(examples: Sequence[Example], preprocess: Preprocess, batch_size: int,
                 num_workers: int, pin_memory: bool = False) -> Iterable[Batch]:
    return torch.utils.data.DataLoader(ImageTextDataset(examples), batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers,
                                       collate_fn=preprocess, pin_memory=pin_memory)


def load_clip(spec: CacheSpec, device: torch.device, dtype: torch.dtype):
    """Frozen ``CLIPModel`` in eval mode plus its ``Preprocess``."""
    from transformers import AutoModel, AutoProcessor, CLIPModel

    loader = AutoModel if spec.family == "siglip" else CLIPModel
    model = loader.from_pretrained(spec.backbone, dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(spec.backbone)
    image_processor = processor.image_processor
    crop = (image_processor.size if spec.family == "siglip"  # SigLIP resizes, no crop
            else image_processor.crop_size)  # a dict before transformers 5, a SizeDict after
    size = crop["height"] if isinstance(crop, dict) else crop.height
    if size != spec.image_size:
        raise ValueError(f"{spec.backbone} processes {size}px images, expected {spec.image_size}")
    return model, Preprocess(processor.image_processor, processor.tokenizer, spec.pre_resize,
                             spec.max_text_tokens)


def default_resize_filter() -> str:
    """Name of the filter ``PIL.Image.resize`` uses for RGB when none is given."""
    from PIL import Image

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))
    default = np.asarray(image.resize((19, 23)))
    for resample in Image.Resampling:
        if np.array_equal(np.asarray(image.resize((19, 23), resample=resample)), default):
            return resample.name
    raise RuntimeError("Pillow's default resize filter matches no Image.Resampling member")


def resize_meta(spec: CacheSpec) -> dict:
    """Manifest fields for the pre-processor resize (#13 parity) and the Pillow version."""
    import PIL

    return {"pre_resize": list(spec.pre_resize) if spec.pre_resize else None,
            "pre_resize_filter": default_resize_filter() if spec.pre_resize else None,
            "pil_version": PIL.__version__}


# ── Extraction ─────────────────────────────────────────────────────────────


def _encode_in_chunks(encode: Encode, batch: Batch, chunk: int) -> tuple[dict, int]:
    """Encode a batch in sub-batches of ``chunk``, halving it on CUDA out-of-memory."""
    _, pixels, input_ids, mask, _ = batch
    while True:
        try:
            parts = [encode(pixels[s:s + chunk], input_ids[s:s + chunk], mask[s:s + chunk])
                     for s in range(0, len(pixels), chunk)]
            return {k: torch.cat([p[k].cpu() for p in parts]) for k in OUTPUT_KEYS}, chunk
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            torch.cuda.empty_cache()
            chunk //= 2
            print(f"CUDA out of memory: batch size halved to {chunk}")


def extract_split(examples: Sequence[Example], batches: Iterable[Batch], encode: Encode,
                  out_dir: str | Path, *, meta: dict, batch_size: int) -> Path:
    """Encode every example once and write one cache directory with ``write_cache``.

    Rows are buffered in memory-mapped scratch files next to ``out_dir`` (removed
    afterwards), so only one split's arrays are ever fully in RAM, inside write_cache.
    """
    out_dir = Path(out_dir)
    scratch = out_dir.parent / f".{out_dir.name}.partial"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    n, chunk, truncated, start = len(examples), batch_size, 0, time.time()
    buffers: dict[str, np.ndarray] = {}
    try:
        for batch in batches:
            out, chunk = _encode_in_chunks(encode, batch, chunk)
            rows = batch[0].numpy()
            truncated += batch[4]
            for key, value in out.items():
                keep = key in ("text_mask", "similarity")  # bool and float32; the rest is fp16
                array = value.numpy() if keep else value.half().numpy()
                if key not in buffers:
                    buffers[key] = np.lib.format.open_memmap(
                        scratch / f"{key}.npy", mode="w+", dtype=array.dtype,
                        shape=(n, *array.shape[1:]))
                buffers[key][rows] = array
        seconds = time.time() - start
        labels = np.array([UNLABELED if e.label is None else e.label for e in examples])
        meta = {**meta, "batch_size": batch_size, "batch_size_used": chunk,
                "texts_truncated": truncated, "extract_seconds": round(seconds, 1)}
        return write_cache(out_dir, ids=[e.id for e in examples], labels=labels, meta=meta,
                           **{key: buffers[key] for key in OUTPUT_KEYS})
    finally:
        buffers.clear()
        shutil.rmtree(scratch, ignore_errors=True)


def spot_check(cache_dir: str | Path, *, n: int = 100, device: torch.device | None = None,
               batch_size: int = 16, clip: tuple | None = None) -> dict:
    """Compare a cache's pooled rows with fp32 CLIP on its first ``n`` samples (D-020).

    ``v1`` is checked against ``get_image_features`` / ``get_text_features``; a
    pooled-only cache against the pre-projection pooler outputs it stores. ``passed``
    applies D-020: per-row ‖Δ‖₂/‖ref‖₂ ≤ ``MAX_REL`` and cosine ≥ ``MIN_COS`` for both
    pooled vectors. ``clip`` reuses an already loaded fp32 ``(model, preprocess)``.
    """
    cache = load_cache(cache_dir)
    manifest = cache.manifest
    spec = SPECS[manifest["cache_version"]]
    examples = load_source(manifest["dataset"], manifest["source_root"], manifest["split"])[:n]
    if [e.id for e in examples] != cache.ids[:n]:
        raise ValueError(f"{cache_dir}: rows do not match the source split's order")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip or load_clip(spec, device, torch.float32)
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("the spot-check reference must be fp32 CLIP")
    image_ref, text_ref = [], []
    with torch.inference_mode():
        for _, pixels, input_ids, mask, _ in make_batches(examples, preprocess, batch_size, 0):
            pixels, input_ids, mask = pixels.to(device), input_ids.to(device), mask.to(device)
            if spec.pooled_only:
                image_ref.append(model.vision_model(pixel_values=pixels).pooler_output)
                text_ref.append(model.text_model(input_ids=input_ids,
                                                 attention_mask=mask).pooler_output)
            else:
                image_ref.append(pooler(model.get_image_features(pixel_values=pixels)))
                text_mask = None if spec.family == "siglip" else mask  # as encoded
                text_ref.append(pooler(model.get_text_features(input_ids=input_ids,
                                                               attention_mask=text_mask)))
    result = {"n": len(examples), "reference": "fp32"}
    for name, ref, cached in (("image", image_ref, cache.image_pooled),
                              ("text", text_ref, cache.text_pooled)):
        ref = torch.cat(ref).float().cpu().numpy()
        got = np.asarray(cached[:len(examples)], dtype=np.float32)
        result.update({f"{name}_{k}": v for k, v in pooled_errors(ref, got).items()})
    result["passed"] = all(result[f"{name}_max_rel"] <= MAX_REL and
                           result[f"{name}_min_cos"] >= MIN_COS for name in ("image", "text"))
    return result


def pooled_errors(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    """Max absolute error, max per-row ‖ref − got‖₂ / ‖ref‖₂ and min cosine over rows."""
    ref_norm, got_norm = np.linalg.norm(ref, axis=1), np.linalg.norm(got, axis=1)
    return {"max_abs": float(np.abs(ref - got).max()),
            "max_rel": float((np.linalg.norm(ref - got, axis=1) / ref_norm).max()),
            "min_cos": float(((ref * got).sum(1) / (ref_norm * got_norm)).min())}


def _git_dirty() -> bool | None:
    proc = subprocess.run(["git", "diff", "--quiet", "HEAD"], capture_output=True)
    return None if proc.returncode not in (0, 1) else proc.returncode == 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract frozen CLIP feature caches (#11).")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--root", required=True,
                        help="FHM/MAMI release dir; other datasets take ANNOTATIONS:IMAGES")
    parser.add_argument("--roles", nargs="+", default=["train", "val"], choices=ROLES)
    parser.add_argument("--allow-test", action="store_true",
                        help="also cache the test role's source split (features only)")
    parser.add_argument("--cache-version", default="v1", choices=sorted(SPECS))
    parser.add_argument("--out", default="~/features", help="cache root (never the repo)")
    parser.add_argument("--compute-dtype", default="float32", choices=("float32", "float16"),
                        help="CLIP compute precision on CUDA (D-020: float32; CPU always float32)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--check", type=int, default=100,
                        help="spot-check the first N rows of each split against fp32 CLIP")
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.compute_dtype) if device.type == "cuda" else torch.float32
    spec = SPECS[args.cache_version]
    splits = source_splits(args.dataset, args.roles, allow_test=args.allow_test)
    model, preprocess = load_clip(spec, device, dtype)

    def encode(pixels, input_ids, mask):
        with torch.inference_mode():
            return encode_batch(model, spec, pixels.to(device, dtype), input_ids.to(device),
                                mask.to(device))

    import transformers

    base_meta = {
        "backbone": spec.backbone, "cache_version": args.cache_version, "git_sha": git_sha(),
        "git_dirty": _git_dirty(), "dataset": args.dataset, "source_root": args.root,
        "image_size": spec.image_size, "projected": not spec.pooled_only,
        "patch_pool": None if spec.pooled_only else spec.patch_pool,
        "text_context": spec.max_text_tokens, "family": spec.family,
        "compute_dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        **resize_meta(spec),
    }
    failed = []
    for split in splits:
        roles = [r for r in args.roles
                 if source_splits(args.dataset, [r], allow_test=args.allow_test) == [split]]
        examples = load_source(args.dataset, args.root, split)
        out_dir = Path(args.out).expanduser() / args.dataset / args.cache_version / split
        batches = make_batches(examples, preprocess, args.batch_size, args.num_workers,
                               pin_memory=device.type == "cuda")
        meta = {**base_meta, "split": split, "roles": roles, "allow_test": "test" in roles}
        path = extract_split(examples, batches, encode, out_dir, meta=meta,
                             batch_size=args.batch_size)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        print(f"{args.dataset}/{args.cache_version}/{split}: {manifest['n']} rows, "
              f"image {manifest['num_image_tokens']}x{manifest['dim']}, text "
              f"{manifest['max_text_tokens']}x{manifest['text_dim']}, "
              f"{manifest['extract_seconds']} s, batch {manifest['batch_size_used']}, "
              f"{manifest['texts_truncated']} texts truncated -> {path}")
        if args.check:
            clip = (model, preprocess) if dtype == torch.float32 else None
            result = spot_check(path, n=args.check, device=device, clip=clip)
            (path / "spot_check.json").write_text(json.dumps(result, indent=2) + "\n",
                                                  encoding="utf-8")
            print(f"spot check: {result}")
            if not result["passed"]:
                failed.append(split)
    if failed:
        print(f"SPOT CHECK FAILED (D-020) for {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
