| Embeddings | adam100 | adam5000 | graphssl | kNN | Rank |
|---|---|---|---|---|---|
| untrained gin-2L (ogbn_arxiv_20261012/afgrl) | 33.75 ± 3.91 (25.6–38.5) | 53.28 ± 1.35 (51.8–54.4) | 55.14 | 53.26 | 3.8 |
| untrained gcn-3L (bgrl_ogbn_arxiv/bgrl_seed0) | 65.25 ± 0.22 (64.7–65.6) | 69.26 ± 0.05 (69.2–69.3) | 69.50 | 66.70 | 116.1 |
| ogbn_arxiv_20261012/afgrl | 52.26 ± 2.81 (46.0–57.0) | 58.90 ± 0.30 (58.6–59.2) | 58.33 | 58.71 | 23.1 |
| ogbn_arxiv_20261012/barlow_twins | 58.99 ± 0.16 (58.7–59.3) | 62.72 ± 0.09 (62.6–62.8) | 62.76 | 54.38 | 68.6 |
| ogbn_arxiv_20261012/bgrl | 58.69 ± 1.14 (55.6–60.2) | 63.76 ± 0.30 (63.5–64.1) | 63.40 | 59.87 | 49.2 |
| ogbn_arxiv_20261012/dgi | 20.58 ± 7.66 (3.2–29.8) | 48.16 ± 0.25 (47.9–48.3) | 44.13 | 52.87 | 1.5 |
| ogbn_arxiv_20261012/graphcl | 59.20 ± 0.26 (58.6–59.7) | 63.81 ± 0.03 (63.8–63.8) | 63.63 | 54.33 | 38.3 |
| ogbn_arxiv_20261012/graphdino | 54.81 ± 2.34 (50.7–58.5) | 62.46 ± 0.24 (62.2–62.6) | 62.44 | 56.18 | 19.0 |
| ogbn_arxiv_20261012/vicreg | 56.88 ± 0.40 (56.2–57.8) | 61.82 ± 0.03 (61.8–61.9) | 61.94 | 54.17 | 36.8 |
| bgrl_ogbn_arxiv/bgrl_seed0 | 71.13 ± 0.08 (71.0–71.3) | 71.02 ± 0.09 (71.0–71.1) | 70.95 | 68.21 | 166.8 |
| bgrl_ogbn_arxiv/bgrl_seed1 | 71.16 ± 0.07 (71.0–71.3) | 71.27 ± 0.04 (71.2–71.3) | 71.42 | 68.39 | 169.1 |
| bgrl_ogbn_arxiv/bgrl_seed2 | 71.21 ± 0.06 (71.1–71.3) | 71.12 ± 0.06 (71.1–71.2) | 71.31 | 68.56 | 166.8 |
| bgrl_ogbn_arxiv_library_predictor/bgrl_seed0 | 27.28 ± 0.08 (27.2–27.5) | 65.63 ± 0.07 (65.6–65.7) | 67.43 | 64.53 | 157.9 |
| bgrl_ogbn_arxiv_library_predictor/bgrl_seed1 | 24.95 ± 1.04 (22.8–26.4) | 53.96 ± 0.17 (53.8–54.2) | 67.50 | 62.44 | 39.2 |
| bgrl_ogbn_arxiv_library_predictor/bgrl_seed2 | 58.86 ± 0.46 (58.1–59.8) | 68.51 ± 0.11 (68.4–68.6) | 68.81 | 66.81 | 144.1 |

Test accuracy (%) on ogbn-arxiv. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
