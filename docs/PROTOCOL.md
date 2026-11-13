# Evaluation protocol

Rules for splits, selection and reporting. The test split is read once per frozen configuration.

## Rules

1. **Three roles per dataset:**
   - `train`: fitting only.
   - `val`: early stopping, hyperparameter sweep, threshold choice, D1 variant choice, LMM prompt choice.
   - `test`: reporting only.
2. **Test predictions are produced once per frozen config.** `hmf/evaluate.py` must refuse `split=test` unless it gets `--final` and the config hash appears in `results/FINAL_CONFIGS.md`. Each line of `FINAL_CONFIGS.md` holds the hash, the date and the issue number.
3. **Thresholds** for accuracy and F1 are picked on val. FHM maximises accuracy. MAMI, HarMeme and CrisisMMD maximise macro-F1. AUROC needs no threshold. Threshold selection applies to binary tasks only. MVSA-Single and Food-101 use argmax, resolving exact ties by the smallest class index. Macro-F1 averages all C classes, assigning zero when a class F1 denominator is zero. These two datasets use val accuracy for early stopping and per-dataset hyperparameter selection. D1 remains selected once using the existing FHM val AUROC rule. No DeLong test or multiclass AUROC is used for these rows (D-018; #54).
4. **Seeds:** use {0..4} for all methods. Use {0..9} for the headline dynamic-vs-static comparison. Results are mean ± std over seeds.
5. **Hyperparameters:** an Optuna sweep on val with seed 0 and at most 50 trials per (dataset, method family). Freeze the result, then run every seed.
6. **Literature rows** get a † and a citation with page or table number. Add a split note if the source paper used a different split. Never mix reported numbers into significance tests.

## Splits

Sizes below were checked against the source files or papers (details after the table). Anything still marked *(verify)* gets its observed count on download and commit the ID lists to `splits/<dataset>/{train,val,test}.txt`. ID lists are fine to commit; images and text aren't.

| Dataset | train | val (selection) | test (report) | Primary metric | Also report |
|---|---|---|---|---|---|
| FHM, **Phase 1 release** (D-012) | `train.jsonl` (8,500) minus the val carve-out | stratified 10% of train, seed 1234 | `dev.jsonl` = dev_seen (500; 250 hateful) | AUROC | Acc, macro-F1 |
| MAMI Task A | official train (10,000; 5,000 misogynous) minus the val carve-out: 9,000 (4,500) | stratified 10% of train, seed 1234: 1,000 (500) | official test (1,000; 500) | macro-F1 | AUROC, Acc |
| Harm-C (HarMeme V1) | `train.jsonl` (3,013; 1,064 harmful) | `val.jsonl` (177; 61) | `test.jsonl` (354; 124) | Acc, macro-F1 | AUROC |
| Harm-P (HarMeme V1) | `train_v1.jsonl` (2,938; 1,487 harmful + 1 unlabeled row, dropped) | `val_v1.jsonl` (177; 89) | `test_v1.jsonl` (355; 171) | Acc, macro-F1 | AUROC |
| CrisisMMD v2 informative | official agreed-label train (9,601 image–text pairs) | official dev (1,573) | official test (1,534) | Acc, macro-F1 | AUROC |
| MVSA-Single (3 classes) | QMF/MMBT `train.jsonl` (1,555) | `dev.jsonl` (518) | `test.jsonl` (519) | Acc (val selection; argmax) | macro-F1; paired accuracy bootstrap, McNemar correctness |
| Food-101, UPMC (101 classes) | QMF/MMBT `train.jsonl` (62,971) | `dev.jsonl` (5,000) | `test.jsonl` (22,715) | Acc (val selection; argmax) | macro-F1; paired accuracy bootstrap, McNemar correctness |

- **Sources and verified sizes:**
  - **FHM:** the Phase 1 release, as mirrored at `emily49/hateful-memes` (the source supplied by the owner's institution). DrivenData no longer accepts new participants (#3).
    - Its labels are the **pre-relabelling Phase 1 labels**: 823 of 8,500 train labels and 3 of 500 dev labels differ from Phase 2 (Kiela et al. 2021, §3.3). Its counts (train 3,050 hateful / 5,450 not; dev 250 / 250) equal PromptHate Table 1 and Pro-Cap Table 3, so those rows are like-for-like.
    - `test.jsonl` (test_seen) is unlabeled and unused. The Phase 2 counts (Kiela et al. 2021, PMLR 133, Table 1, p. 349) are kept in the loader for reference only.
  - **HarMeme:** V1 annotations from `LCS2-IIITD/MOMENTA` (`HarMeme_V1/Annotations`), with images from the authors' "HarMeme Images" link.
    - Harm-C counts equal PromptHate Table 1 (train 1,064/1,949, test 124/230).
    - Harm-P matches the V1 README except for **one `train_v1.jsonl` row with an empty label**, which the loader drops and logs.
  - **CrisisMMD:** Ofli et al. 2020, ISCRAM, Table 1, p. 5 ("Image" columns). Informative: train 6,345, dev 1,056, test 1,030.
  - **MVSA-Single and Food-101:** the train/dev/test `jsonl` files shipped in QMF's repo (`QingyangZhang/QMF` @ `fe6c4c6`, MMBT splits), with images from the Kaggle copies QMF's README links. These are the splits QMF and PDF report on.
  - **MAMI:** the organisers' release (access granted 2026-10-06). Cite Fersini et al. 2022 as their README requests.
    - Sub-task A counts equal Fersini et al. 2022, SemEval-2022 Task 5, Table 2, p. 536 (train 5,000/5,000; test 500/500), confirmed on the release, whose `test_labels.txt` includes the test labels (#6, PR #91).
    - The release reuses five file names (15001, 15002, 15004, 15005, 15006) in train and test for **different** memes (image bytes and texts differ). Test IDs are therefore `test_<stem>`, so no ID sits in two splits.
    - The files are tab-separated, not quoted CSV, and are read with `QUOTE_NONE` so transcriptions keep their quote characters exactly as released.
  - Counts are encoded as `EXPECTED_COUNTS` in each loader.
- **HarMeme binarisation:** "not harmful" = 0; "somewhat harmful" and "very harmful" = 1 (the label strings in the V1 files). This matches PromptHate (Cao et al. 2022, §5.1, arXiv:2302.04156 p. 5) and Pro-Cap (Cao et al. 2023, §5.1, arXiv:2308.08088 p. 4), which merge "partially" and "very" harmful into the positive class.
- **Multi-class datasets (MVSA-Single, Food-101)** use the owner-approved extension (D-018; #54): softmax experts, reliability targets from per-expert cross-entropy, accuracy and macro-F1, paired bootstrap on accuracy, and McNemar on correctness. DeLong is binary-only and not used for these tasks. See MATH_SPEC §2–§7 and §10 and derivation A-7.
- **Cross-dataset transfer** stays among the binary harm datasets: FHM, MAMI, Harm-C, Harm-P.

## Integrity checks (unit tests, M0)

- No ID appears in more than one split.
- Look for near-duplicate images (dHash, Hamming distance ≤ 4) and **(image, text)** duplicates across splits (D-014):
  - Remove **(image, text)** duplicates from `train` only.
  - **Report** image-only matches, but keep them: FHM's benign confounders reuse images on purpose.
  - Log counts to `splits/<dataset>/dedup_report.json`.
  - Never alter `val` or `test`.
- Check the label distribution of each split against the source paper.
- Every split must have labels, except where an unlabeled test set is explicitly listed as unused.

## Cross-dataset transfer

Train on A with A's val for selection, then score B's test set **with no adaptation**. Report AUROC only, because label semantics differ (hateful vs misogynous vs harmful) and thresholds don't transfer. Datasets: FHM, MAMI, Harm-C, Harm-P. CrisisMMD isn't a harm task, so it stays out of the transfer matrix.

## LMM zero-shot

- Use one prompt template, chosen on FHM val from at most 3 candidates. Freeze it in `configs/lmm_prompt.txt`.
- Score = softmax over the {"Yes", "No"} next-token logits, taking P(Yes). Use greedy decoding, no sampling.
- Log refusals and malformed outputs. Score them 0.5 and report the count.

## Robustness and calibration (pre-registered, D-024; #100)

These conditions are fixed before any main-experiment test run, and nothing is tuned on their results. They are computed **inside the single `hmf/evaluate.py --final` pass** (rule 2), which reads the test split once and writes the clean predictions plus one prediction file per condition.

- **Conditions.** Each one corrupts one modality of the cached features at evaluation only; training never changes:
  - **drop-image / drop-text:** every token of that modality becomes the train-role mean token. The text mask is unchanged.
  - **noise-image / noise-text, σ ∈ {0.5, 1.0}:** add N(0, (σ·s_d)²) independently per feature dimension d, where s_d is the train-role standard deviation of dimension d.
  - **swap-image / swap-text:** that modality comes from another test sample, via one fixed random permutation (seed 1234) with no fixed points.
- **Statistics** (mean and s_d) come from the train role's cache only. Randomness uses seed 1234.
- **Methods:** every method with a test run in #27, under the same conditions.
- **Metrics:** AUROC, plus accuracy and macro-F1 at the clean val threshold, or argmax for multi-class. For routed models, also the mean π per condition.
- **Calibration:** expected calibration error with 15 equal-width bins. Binary uses P(y=1); multi-class uses the top-label confidence. It is reported for val and the clean test.
- **Router evidence:** per-expert probabilities are written alongside π in every prediction file. The per-sample reliability target t (MATH_SPEC §5) is computed from them for analysis only.

## Mixed-quality inputs (pre-registered, D-027; #152)

A separate confirmatory experiment. It is fixed before any run, and nothing is tuned on its results.

- **Hypothesis H-MQ.** When one modality is missing on some samples but not others, dynamic fusion beats static fusion. On a sample whose modality is missing, the router gives that modality less weight than it does on intact samples.
- **Corruption rule (every role: train, val and test).** For each sample, u = the first 8 bytes of SHA-256(`"1234:<dataset>:<id>"`) as an integer / 2^64.
  - u < 0.25: image missing. Every image token and the image pooled vector become the train-role mean image token (the D-024 drop).
  - 0.25 ≤ u < 0.5: text missing. The same with the text mean token; the mask is unchanged.
  - Otherwise the sample is intact. The rule is fixed per sample, so it is identical across epochs, seeds and methods. Statistics come from the clean train role.
- **Datasets:** FHM, MAMI, Harm-C, Harm-P, CrisisMMD and MVSA-Single, all six, none dropped. Food-101 is excluded for compute.
- **Configs:** each dataset's frozen clean config per method (#20, D-026), plus the corruption key. There is no new sweep. Val selection (epoch, threshold) works as usual on the corrupted val role.
- **Methods and seeds:** dynamic and static, seeds 0–9; global, seeds 0–4; QMF and PDF, seeds 0–4 once their clean configs are frozen.
- **Test:** one `--final` pass per run on the corrupted test role, after owner approval in FINAL_CONFIGS. The D-024 conditions are not applied on top.
- **Primary analysis:** MATH_SPEC §10.1 (paired t-test across 10 seeds, test AUROC; test accuracy for MVSA), with Holm–Bonferroni over the six datasets. Secondary analyses are DeLong, the paired bootstrap and McNemar per seed, and the router evidence: mean π per expert on image-missing, text-missing and intact test samples.
- **Reporting:** all six datasets, whatever the outcome. The clean headline results (#27) are reported unchanged.

