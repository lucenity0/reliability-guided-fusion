Mean test AUROC over seeds. Off-diagonal rows score a model on another dataset's test set with no adaptation; the diagonal (same dataset) is each model's own result. MAMI-trained dynamic/static models are a 5-seed retrain (the originals were lost).

| Train → test | Dynamic | Static fusion | QMF | PDF | Concat MLP |
|---|---|---|---|---|---|
| FHM → FHM | 0.756 | 0.756 | 0.731 | 0.739 | 0.760 |
| FHM → MAMI | 0.693 | 0.694 | 0.689 | 0.687 | 0.698 |
| FHM → Harm-C | 0.508 | 0.508 | 0.517 | 0.515 | 0.553 |
| FHM → Harm-P | 0.507 | 0.505 | 0.489 | 0.486 | 0.483 |
| MAMI → FHM | 0.539 | 0.534 | 0.532 | 0.533 | 0.547 |
| MAMI → MAMI | 0.852 | 0.847 | 0.862 | 0.859 | 0.863 |
| MAMI → Harm-C | 0.368 | 0.410 | 0.370 | 0.389 | 0.384 |
| MAMI → Harm-P | 0.531 | 0.521 | 0.515 | 0.525 | 0.527 |
| Harm-C → FHM | 0.604 | 0.584 | 0.579 | 0.589 | 0.585 |
| Harm-C → MAMI | 0.669 | 0.670 | 0.683 | 0.679 | 0.675 |
| Harm-C → Harm-C | 0.923 | 0.916 | 0.924 | 0.923 | 0.919 |
| Harm-C → Harm-P | 0.604 | 0.595 | 0.584 | 0.606 | 0.604 |
| Harm-P → FHM | 0.606 | 0.606 | 0.598 | 0.597 | 0.613 |
| Harm-P → MAMI | 0.675 | 0.667 | 0.640 | 0.625 | 0.711 |
| Harm-P → Harm-C | 0.675 | 0.652 | 0.645 | 0.657 | 0.686 |
| Harm-P → Harm-P | 0.679 | 0.679 | 0.668 | 0.667 | 0.718 |
