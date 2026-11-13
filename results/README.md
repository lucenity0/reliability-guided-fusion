# Results

Layout (stable interface, see the design notes §7):

```
results/<dataset>/<method>/seed<k>/
  preds_val.csv     binary: id,label,prob[,pi_img,pi_txt,pi_int]  (prob = P(y=1))
                    multi-class: id,label,pred,prob[,pi_*]          (argmax and its probability)
  preds_test.csv    only written by `hmf.evaluate --final` (docs/PROTOCOL.md)
  metrics.json      includes git_sha, config_hash, best_epoch, train_seconds, device;
                    val (and test) metrics, the val-chosen threshold, router health, history
  config.yaml       exact config used
results/FINAL_CONFIGS.md   config hashes cleared for test evaluation (hash, date, issue)
```

Prediction CSVs contain IDs and scores only, never images or text. They are omitted from this public copy because of their size (about 350 MB); `metrics.json` and `config.yaml` are kept for every run.
`hmf/report.py` must regenerate every paper table and figure from this directory.

## Final-pass outputs (D-024, #100)

The existing `python -m hmf.evaluate --run-dir ... --split test --final`
command writes `preds_test.csv` and eight `preds_test_<condition>.csv` files:
`drop-image`, `drop-text`, `noise-image-0.5`, `noise-text-0.5`,
`noise-image-1.0`, `noise-text-1.0`, `swap-image`, and `swap-text`.
All reuse the frozen clean-val threshold (binary) or argmax (multi-class).
There is no new config key, hash change, or test selection step.

Every prediction export, including training's val/epoch files, appends
`p_<expert>` for binary expert probabilities, or `p_<expert>_<class index>`
for every class in a categorical expert. Existing leading columns stay in
order. Expert probabilities use nine significant digits. Unimodal models
export their single expert; V3, QMF and PDF export img/txt; dynamic, static
and global export img/txt/int. Models without expert heads (such as concat)
have no expert columns. QMF retains its existing absence of simplex weights.

`metrics.json` includes:

- `val.calibration` and `test.calibration`: `ece`, `n_bins` (15), `kind`,
  and `bins`, a reliability-diagram table with bin edges, count, mean probability
  and observed frequency. Binary observed frequency is the fraction of positive
  labels; categorical observed frequency is top-label accuracy. Bins are
  left-closed/right-open, with 1 included in the final bin. Empty bin means are null.
- `test.corruptions.<condition>`: accuracy, macro-F1, binary AUROC/threshold,
  calibration, row count, and `pi_mean` when available. Clean `pi_mean` is under `test`.
- `test.robustness`: seed 1234, the train-role ID-list provenance and valid-token
  counts used for statistics, and swap donor IDs in prediction-row order.

Statistics stream only the train-role IDs, respecting carve-outs and dedup.
The per-dimension population mean/std treats all image tokens equally and
excludes masked text padding. Drop fills both tokens and pooled vectors;
noise reuses the CLS/EOS token's perturbation in its pooled vector. Text swaps
carry the donor mask. Cosine similarity follows the corrupted pooled vectors
for projected caches; pre-projection Hate-CLIPper caches have incompatible
dimensions and their models do not consume similarity. Noise is keyed by row
and modality, independent of model seed, batch size or access order; the two
sigmas scale the same standardized noise. Both swaps use the same seeded
derangement, with no fixed points.

The final command refreshes an older run's val export and calibration using
its saved checkpoint and threshold. Train/val ID hashes, when recorded at
training, must still match. It then opens the test cache once and streams its
rows once to a temporary disk snapshot for random-access swaps. Feature
batches are capped at 16 and workers at zero. Set `TMPDIR` to a disk with space
for the test features in float32 (roughly twice the fp16 cache footprint);
the snapshot is removed even on failure. Total RAM also depends on model size.

For #32, `hmf.eval.router_evidence.router_evidence(csv_path, expert_names,
num_classes=C, tau=frozen_tau)` computes per-sample targets from expert CE,
per-expert Pearson correlation with pi, argmax agreement, and mean pi from
any exported condition. Targets follow CSV row order. Constant-vector
correlations are null; argmax ties use the supplied expert order. Exact zero
likelihoods are floored at float64's smallest positive normal for logs. This
is post-hoc analysis only and never affects predictions or selection.
