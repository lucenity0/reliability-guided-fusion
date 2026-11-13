# Showcase: an interactive paper

This is a local web page that tells the project as a scrolling paper. It is styled as ink on matte paper, with a dark "inverted e-ink" mode. Figures stay pinned and build step by step as you read:

1. **The problem:** an invented, drawn example of when to trust the image, the text, or both.
2. **Why fixed weights fail:** V3's gate collapsing to α = 0.5.
3. **Method:** the architecture assembles itself, with equations (1)–(4).
4. **Try it:** the live Food-101 model. It shows the prediction, the router weights π and each expert's view, and runs a four-way stress test: original, image removed, text removed, mismatched text.
5. **Findings:** the π triangle for clean FHM, then clean Harm-C, then Harm-C under mixed quality (animated), a slope chart, and the significance plot. After it comes an interactive explorer covering all seven datasets.
6. **Limits:** transfer, ablations and cost.
7. **Full results table.**

## Run

```bash
pip install -e ".[train]"                # torch, transformers, pyyaml, pillow (same as training)
python -m showcase.app                   # opens http://127.0.0.1:8765
python -m showcase.app --explorer-only   # no model download; every section except the live demo
```

The first launch downloads CLIP ViT-L/14@336 (about 1.7 GB) into the Hugging Face cache. After that the app works offline. Loading takes about 10 s. One prediction (all four inputs) takes about 0.6–1.8 s on an Apple-silicon GPU.

## What it runs

- **Features:** `hmf.features.extract.encode_batch` on frozen CLIP. This is the code that built the training caches: fp32 compute, with tokens stored as fp16.
- **Models:** `app/models/food101/{dynamic,static}/best.pt` with the exact `config.yaml`. Their config hashes match `results/food101/*/seed0/metrics.json`.
- **Data for §5–§7:** `app/static/data/explorer.json`, regenerated with `python -m showcase.export_explorer_data` from `results/` and `docs/tables/` only (the regeneration needs the per-sample prediction files, which are not part of this public copy; the generated `explorer.json` is). It holds IDs and scores, never dataset images or text.

## Files

| File | Purpose |
|---|---|
| `app/static/index.html` | Text and structure of the paper |
| `app/static/paper.css` | Design tokens (paper and ink, both themes), layout, patterns |
| `app/static/paper.js` | Scrollytelling, figures, live instrument, explorer, table |
| `app/static/progress.js` | Optional reading-progress rail; delete its `<script>` tag to switch it off |

## Fonts

The display face is Mogaly, a licensed Envato font that isn't committed. Put `Mogaly.otf` in `app/static/fonts/`; without it, headings fall back to Charter. Body text uses Charter, which ships with macOS.
