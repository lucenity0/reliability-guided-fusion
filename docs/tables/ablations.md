FHM validation set, seeds 0–4, every variant trained in the same session; p from the paired t-test across seeds (uncorrected).

| Variant | Val AUROC | Δ vs full | p |
|---|---|---|---|
| Full model (tuned, τ = 4) | 0.8339 ± 0.0029 | – | – |
| τ = 0.25 | 0.8337 ± 0.0037 | -0.0002 | 0.809 |
| τ = 0.5 | 0.8340 ± 0.0030 | +0.0001 | 0.773 |
| τ = 1 | 0.8339 ± 0.0030 | +0.0000 | 0.825 |
| τ = 2 | 0.8339 ± 0.0029 | +0.0000 | 0.705 |
| λ_rel = 0 (no reliability target) | 0.8281 ± 0.0052 | -0.0058 | 0.017 |
| Without image expert | 0.8256 ± 0.0032 | -0.0082 | 0.002 |
| Without text expert | 0.8315 ± 0.0053 | -0.0023 | 0.279 |
| Without interaction expert | 0.8249 ± 0.0023 | -0.0089 | 0.002 |
| Router without entropy | 0.8339 ± 0.0029 | +0.0001 | 0.304 |
| Router without similarity | 0.8339 ± 0.0028 | +0.0000 | 0.557 |
| Router without pooled embeddings | 0.8326 ± 0.0030 | -0.0013 | 0.053 |
