Device: NVIDIA L4. Fusion heads on cached features; the backbone row is the frozen CLIP forward.

| Method | Params (M) | FLOPs (M) | ms/sample (batch 1) | ms/sample (batch 64) | Peak MB (batch 64) | Train s/seed (MAMI) |
|---|---|---|---|---|---|---|
| Image only | 0.526 | 57.9 | 0.626 | 0.010 | 71 | 245 |
| Text only | 0.526 | 31.2 | 0.947 | 0.012 | 78 | 118 |
| Concat MLP | 0.394 | 0.8 | 0.192 | 0.004 | 53 | 104 |
| Linear probe | 0.002 | 0.0 | 0.254 | 0.004 | 52 | 309 |
| V3 scalar-α | 2.842 | 5.7 | 1.456 | 0.023 | 64 | 86 |
| QMF | 3.414 | 180.4 | 1.657 | 0.027 | 101 | 234 |
| PDF | 5.515 | 184.6 | 2.135 | 0.033 | 109 | 316 |
| Static fusion | 3.931 | 601.7 | 4.656 | 0.136 | 168 | 383 |
| Global-π | 3.931 | 601.7 | 4.664 | 0.138 | 168 | 372 |
| Dynamic (ours) | 4.459 | 602.8 | 5.068 | 0.137 | 170 | 303 |
| CLIP ViT-L/14@336 (frozen) | 427.9 | 362270 | 62.53 | 57.30 | 4036 | – |
| LLaVA-1.6 7B (0-shot) | – | – | 3782 | – | – | – |
| Qwen2.5-VL 7B (0-shot) | – | – | 1569 | – | – | – |
