# ruff: noqa: E501
"""Local demo server: live Food-101 predictions plus the research explorer.

    python -m showcase.app            # http://127.0.0.1:8765

The live demo runs the exact feature path used to build the training caches
(hmf.features.extract.encode_batch on frozen CLIP ViT-L/14@336, fp32 compute,
fp16 storage) and the frozen Food-101 checkpoints (seed 0) of the dynamic
model and of static fusion. Nothing here trains or selects anything.
Standard library HTTP server only; no web framework needed.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP = Path(__file__).resolve().parent
STATIC = APP / "static"
MODELS = APP / "models" / "food101"
EXPERTS = ("img", "txt", "int")

# Mismatched-text probe: a description of a clearly different dish. The first
# entry whose class differs from the original prediction is used.
MISMATCH_POOL = [
    ("sushi", "Fresh salmon and tuna nigiri sushi with pickled ginger, wasabi and soy sauce."),
    ("chocolate_cake", "Rich triple-layer chocolate cake with dark ganache frosting."),
    ("caesar_salad", "Crisp romaine lettuce, parmesan, croutons and creamy Caesar dressing."),
    ("pizza", "Neapolitan pizza with tomato sauce, fresh mozzarella and basil leaves."),
]


class Engine:
    """Loads CLIP and both fusion heads once; answers predict() calls."""

    def __init__(self):
        self.status, self.error, self.device = "loading", None, None
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            import torch
            import yaml

            from hmf.config import get_device
            from hmf.data.food101 import CLASSES
            from hmf.features.extract import SPECS, load_clip
            from hmf.models.registry import build_model
            from hmf.utils.checkpointing import CheckpointManager

            self.torch, self.classes = torch, CLASSES
            self.device = get_device()
            self.spec = SPECS["v1"]
            t0 = time.time()
            self.clip, self.preprocess = load_clip(self.spec, self.device, torch.float32)  # D-020
            self.models, self.meta = {}, {}
            for name in ("dynamic", "static"):
                cfg = yaml.safe_load((MODELS / name / "config.yaml").read_text())
                model = build_model(cfg).to(self.device).eval()
                CheckpointManager(MODELS / name).load_best(model, map_location=self.device)
                self.models[name] = model
                metrics = json.loads((MODELS / name / "metrics.json").read_text())
                self.meta[name] = {"test_accuracy": metrics.get("test", {}).get("accuracy"),
                                   "config_hash": metrics["config_hash"]}
            from PIL import Image
            self.blank = Image.new("RGB", (336, 336), (127, 127, 127))
            self.predict(self.blank, "warm-up")  # first MPS/CUDA call compiles kernels (~5 s)
            self.load_seconds = round(time.time() - t0, 1)
            self.status = "ready"
        except Exception as exc:  # surfaced to the UI through /api/status
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def _features(self, pairs):
        """Cache-format feature batch for [(PIL image, text)], as hmf.features builds it."""
        torch = self.torch
        rows = list(range(len(pairs)))
        _, pixels, input_ids, mask, _ = self.preprocess(
            [(r, img, txt) for r, (img, txt) in zip(rows, pairs, strict=True)])
        with torch.inference_mode():
            enc = __import__("hmf.features.extract", fromlist=["encode_batch"]).encode_batch(
                self.clip, self.spec, pixels.to(self.device), input_ids.to(self.device),
                mask.to(self.device))
        length = int(enc["text_mask"].sum(dim=1).max())

        def stored(t):  # the cache keeps tokens in fp16 (MATH_SPEC §1)
            return t.half().float()

        image = stored(enc["image_tokens"])
        return {"image_tokens": image, "image_pooled": image[:, 0],
                "text_tokens": stored(enc["text_tokens"][:, :length]),
                "text_mask": enc["text_mask"][:, :length].to(self.device),
                "text_pooled": stored(enc["text_pooled"]),
                "similarity": enc["similarity"].float()}

    def _top(self, probs, k=5):
        values, idx = probs.topk(k)
        return [{"label": self.classes[i], "p": round(float(v), 4)}
                for v, i in zip(values.tolist(), idx.tolist(), strict=True)]

    def _run(self, pairs):
        torch = self.torch
        batch = self._features(pairs)
        with torch.inference_mode():
            dyn = self.models["dynamic"](batch)
            sta = self.models["static"](batch)
        out = []
        for i in range(len(pairs)):
            z = dyn["z"][i].float()                         # [K, C] expert logits
            p_k = z.softmax(dim=-1)
            ent = -(p_k * p_k.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(float(z.shape[-1])))
            out.append({
                "dynamic": self._top(dyn["prob"][i].float()),
                "static": self._top(sta["prob"][i].float()),
                "pi": [round(float(v), 4) for v in dyn["pi"][i].tolist()],
                "experts": [{"name": e, "top": self._top(p_k[j], 3), "entropy": round(float(ent[j]), 4)}
                            for j, e in enumerate(EXPERTS)],
                "similarity": round(float(batch["similarity"][i]), 4),
            })
        return out

    def predict(self, image, text: str) -> dict:
        t0 = time.time()
        with self.lock:
            first = self._run([(image, text), (self.blank, text), (image, "")])
            top = first[0]["dynamic"][0]["label"]
            mismatch = next(s for c, s in MISMATCH_POOL if c != top)
            second = self._run([(image, mismatch)])
        conditions = [
            {"key": "original", "title": "Original", "detail": "your image and text", **first[0]},
            {"key": "blank_image", "title": "Image removed",
             "detail": "image replaced by a blank grey canvas", **first[1]},
            {"key": "blank_text", "title": "Text removed", "detail": "text replaced by an empty string",
             **first[2]},
            {"key": "mismatch", "title": "Mismatched text", "detail": mismatch, **second[0]},
        ]
        return {"conditions": conditions, "ms": round((time.time() - t0) * 1000)}


ENGINE = Engine()


class Handler(BaseHTTPRequestHandler):
    server_version = "RGF-Demo/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/status":
            return self._json(200, {"status": ENGINE.status, "error": ENGINE.error,
                                    "device": str(ENGINE.device) if ENGINE.device else None,
                                    "meta": getattr(ENGINE, "meta", None),
                                    "load_seconds": getattr(ENGINE, "load_seconds", None)})
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        file = (STATIC / rel).resolve()
        if STATIC not in file.parents and file != STATIC / "index.html" or not file.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        self._send(200, file.read_bytes(), ctype)

    def do_POST(self):  # noqa: N802
        if self.path != "/api/predict":
            return self._send(404, b"not found", "text/plain")
        if ENGINE.status != "ready":
            return self._json(503, {"error": f"model {ENGINE.status}", "detail": ENGINE.error})
        try:
            from PIL import Image

            length = int(self.headers.get("Content-Length", 0))
            if length > 25_000_000:
                return self._json(413, {"error": "image too large (max ~18 MB)"})
            body = json.loads(self.rfile.read(length))
            data = body["image"].split(",", 1)[-1]
            image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
            text = str(body.get("text", ""))[:2000]
            self._json(200, ENGINE.predict(image, text))
        except Exception as exc:
            traceback.print_exc()
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--explorer-only", action="store_true", help="skip loading the model")
    args = parser.parse_args(argv)
    mimetypes.add_type("font/ttf", ".ttf")
    mimetypes.add_type("font/otf", ".otf")
    mimetypes.add_type("font/woff2", ".woff2")
    if args.explorer_only:
        ENGINE.status = "disabled"
    else:
        threading.Thread(target=ENGINE.load, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Reliability-Guided Fusion demo at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
