# Reported literature numbers (†-rows)

`literature_numbers.csv` holds every published number we may quote. Each value was checked against the source PDF on 2026-10-08. `location` gives the table and the **PDF page**, except for Hate-CLIPper, whose pages are the printed proceedings pages (171–183).

These rows are **reported only**. They never enter significance tests (PROTOCOL rule 6). In a paper table they get a † and the citation.

## How comparable each row is with our protocol (D-012)

| Dataset | Like-for-like | Caveat |
|---|---|---|
| FHM | PromptHate and Pro-Cap: same Phase 1 labels, and dev_seen as test | None of them, nor Kiela et al. 2020, describe a validation split separate from dev_seen. Our protocol is stricter (10% carve-out). Hate-CLIPper doesn't say which label release it used, and tuned on "the validation set". |
| Harm-C | PromptHate, Pro-Cap, MOMENTA | V0 and V1 are identical for Harm-C. |
| Harm-P | n/a | MOMENTA reports on V0, which contains duplicates. We use deduplicated V1, so treat it as a reference point only. |
| MAMI | Pro-Cap (AUROC/Acc), and the organisers' leaderboard (macro-F1) | Leaderboard entries are competition systems, often ensembles. |
| MVSA-Single, Food-101 | QMF and PDF on the same MMBT splits | Their encoders differ from our frozen CLIP. Only our re-implementations on identical features (#23) support a method comparison. |
| CrisisMMD | Ofli et al. 2020 on the same agreed-label split | The averaging behind their F1 isn't stated in the table. |

Not yet collected (the related-work pass, #34): Mod-HATE and 2024–2026 LMM-based detectors. Add them only with a table and page, and with their split stated.
