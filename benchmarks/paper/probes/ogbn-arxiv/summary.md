| Embeddings | adam100 | adam5000 | graphssl | kNN | Rank |
|---|---|---|---|---|---|
| transformer/afgrl_seed0 | 67.63 ± 0.12 (67.4–68.0) | 68.87 ± 0.06 (68.8–68.9) | 68.97 | 66.57 | 59.6 |
| transformer/afgrl_seed1 | 67.62 ± 0.16 (67.3–67.9) | 69.09 ± 0.05 (69.0–69.1) | 69.08 | 66.40 | 61.5 |
| transformer/afgrl_seed2 | 67.60 ± 0.13 (67.4–67.8) | 69.11 ± 0.11 (69.0–69.2) | 69.09 | 66.58 | 54.0 |
| transformer/barlow_twins_seed0 | 57.68 ± 0.15 (57.4–58.0) | 63.09 ± 0.08 (63.0–63.2) | 62.90 | 46.95 | 107.1 |
| transformer/barlow_twins_seed1 | 57.52 ± 0.13 (57.2–57.7) | 63.24 ± 0.06 (63.2–63.3) | 63.22 | 47.24 | 108.8 |
| transformer/barlow_twins_seed2 | 57.38 ± 0.15 (57.1–57.6) | 62.92 ± 0.05 (62.9–63.0) | 63.03 | 46.47 | 108.0 |
| transformer/bgrl_seed0 | 59.48 ± 0.32 (58.4–59.9) | 61.93 ± 0.06 (61.9–62.0) | 61.68 | 46.95 | 148.5 |
| transformer/bgrl_seed1 | 59.69 ± 0.19 (59.3–60.0) | 61.95 ± 0.04 (61.9–62.0) | 61.95 | 47.27 | 145.8 |
| transformer/bgrl_seed2 | 59.02 ± 0.23 (58.4–59.4) | 61.84 ± 0.04 (61.8–61.9) | 61.81 | 46.79 | 146.6 |
| transformer/dgi_seed0 | 34.77 ± 2.00 (30.6–38.5) | 65.18 ± 0.55 (64.6–65.6) | 65.03 | 53.24 | 87.4 |
| transformer/dgi_seed1 | 34.49 ± 2.10 (28.0–37.8) | 64.12 ± 1.10 (62.9–65.0) | 65.34 | 55.36 | 85.1 |
| transformer/dgi_seed2 | 35.12 ± 1.43 (33.3–39.2) | 64.12 ± 1.59 (62.5–65.7) | 65.05 | 54.00 | 75.2 |
| transformer/graphcl_seed0 | 57.52 ± 0.14 (57.3–57.8) | 62.87 ± 0.13 (62.7–63.0) | 63.00 | 47.05 | 114.2 |
| transformer/graphcl_seed1 | 57.50 ± 0.12 (57.2–57.7) | 62.70 ± 0.19 (62.5–62.9) | 62.56 | 47.34 | 116.6 |
| transformer/graphcl_seed2 | 57.56 ± 0.17 (57.2–58.0) | 63.04 ± 0.03 (63.0–63.1) | 63.04 | 46.73 | 115.9 |
| transformer/graphdino_seed0 | 60.88 ± 0.11 (60.7–61.2) | 63.26 ± 0.04 (63.2–63.3) | 63.17 | 49.13 | 125.6 |
| transformer/graphdino_seed1 | 61.00 ± 0.10 (60.8–61.2) | 63.37 ± 0.03 (63.3–63.4) | 63.47 | 49.20 | 125.2 |
| transformer/graphdino_seed2 | 61.37 ± 0.15 (61.1–61.6) | 63.38 ± 0.07 (63.3–63.5) | 63.45 | 49.31 | 124.6 |
| transformer/vicreg_seed0 | 57.29 ± 0.15 (57.0–57.5) | 62.84 ± 0.09 (62.7–62.9) | 62.73 | 46.57 | 113.5 |
| transformer/vicreg_seed1 | 57.35 ± 0.10 (57.2–57.6) | 62.66 ± 0.16 (62.5–62.8) | 62.55 | 47.03 | 112.9 |
| transformer/vicreg_seed2 | 57.00 ± 0.13 (56.8–57.2) | 62.15 ± 0.12 (62.0–62.3) | 62.26 | 46.43 | 114.7 |
| transformer/bgrl_seed0 | 54.28 ± 0.18 (54.0–54.6) | 65.84 ± 0.11 (65.8–66.0) | 66.59 | 52.47 | 153.2 |
| transformer/bgrl_seed1 | 54.49 ± 0.26 (54.0–54.9) | 65.86 ± 0.10 (65.8–65.9) | 66.28 | 52.35 | 153.8 |
| transformer/bgrl_seed2 | 54.75 ± 0.26 (54.4–55.3) | 65.76 ± 0.07 (65.7–65.8) | 66.41 | 52.53 | 153.9 |

Test accuracy (%) on ogbn-arxiv. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
