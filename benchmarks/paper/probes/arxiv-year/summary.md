| Embeddings | adam100 | adam5000 | graphssl | kNN | Rank |
|---|---|---|---|---|---|
| transformer/afgrl_seed0 | 37.23 ± 0.33 (36.2–37.7) | 40.19 ± 0.03 (40.2–40.2) | 40.14 | 37.70 | 59.6 |
| transformer/afgrl_seed1 | 37.13 ± 0.33 (36.4–37.6) | 40.10 ± 0.04 (40.1–40.1) | 40.19 | 37.73 | 61.5 |
| transformer/afgrl_seed2 | 37.51 ± 0.22 (37.1–37.8) | 40.38 ± 0.08 (40.3–40.5) | 40.42 | 38.10 | 54.0 |
| transformer/barlow_twins_seed0 | 37.07 ± 0.10 (36.8–37.2) | 37.73 ± 0.11 (37.6–37.8) | 37.80 | 29.77 | 107.1 |
| transformer/barlow_twins_seed1 | 37.09 ± 0.11 (36.9–37.3) | 37.95 ± 0.12 (37.8–38.1) | 37.82 | 29.94 | 108.8 |
| transformer/barlow_twins_seed2 | 36.98 ± 0.15 (36.7–37.2) | 37.68 ± 0.04 (37.6–37.7) | 37.71 | 29.77 | 108.0 |
| transformer/bgrl_seed0 | 36.68 ± 0.16 (36.4–37.0) | 37.70 ± 0.01 (37.7–37.7) | 37.71 | 29.69 | 148.5 |
| transformer/bgrl_seed1 | 36.70 ± 0.13 (36.4–37.0) | 37.54 ± 0.04 (37.5–37.6) | 37.56 | 30.05 | 145.8 |
| transformer/bgrl_seed2 | 36.69 ± 0.16 (36.2–36.9) | 37.69 ± 0.03 (37.7–37.7) | 37.59 | 29.55 | 146.6 |
| transformer/dgi_seed0 | 30.43 ± 0.56 (29.3–31.2) | 38.40 ± 0.07 (38.3–38.5) | 38.79 | 34.90 | 87.4 |
| transformer/dgi_seed1 | 31.21 ± 0.88 (30.0–33.0) | 38.71 ± 0.21 (38.5–38.9) | 39.32 | 35.16 | 85.1 |
| transformer/dgi_seed2 | 31.84 ± 0.78 (30.2–33.0) | 38.82 ± 0.05 (38.8–38.9) | 39.00 | 34.96 | 75.2 |
| transformer/graphcl_seed0 | 37.01 ± 0.13 (36.7–37.3) | 37.77 ± 0.02 (37.7–37.8) | 37.85 | 29.62 | 114.2 |
| transformer/graphcl_seed1 | 37.14 ± 0.11 (36.9–37.3) | 37.93 ± 0.04 (37.9–38.0) | 37.77 | 29.73 | 116.6 |
| transformer/graphcl_seed2 | 37.08 ± 0.09 (36.9–37.2) | 37.77 ± 0.02 (37.8–37.8) | 37.74 | 29.92 | 115.9 |
| transformer/graphdino_seed0 | 37.43 ± 0.13 (37.2–37.7) | 37.98 ± 0.01 (38.0–38.0) | 37.91 | 30.58 | 125.6 |
| transformer/graphdino_seed1 | 37.44 ± 0.10 (37.3–37.6) | 38.07 ± 0.05 (38.0–38.1) | 38.03 | 30.44 | 125.2 |
| transformer/graphdino_seed2 | 37.63 ± 0.08 (37.5–37.8) | 38.11 ± 0.11 (38.0–38.2) | 38.15 | 30.52 | 124.6 |
| transformer/vicreg_seed0 | 37.22 ± 0.11 (37.0–37.4) | 37.87 ± 0.08 (37.8–38.0) | 37.74 | 29.87 | 113.5 |
| transformer/vicreg_seed1 | 37.01 ± 0.10 (36.8–37.2) | 37.80 ± 0.01 (37.8–37.8) | 37.84 | 29.78 | 112.9 |
| transformer/vicreg_seed2 | 37.06 ± 0.13 (36.7–37.3) | 37.75 ± 0.06 (37.7–37.8) | 37.69 | 29.69 | 114.7 |
| transformer/bgrl_seed0 | 36.19 ± 0.16 (35.8–36.5) | 38.37 ± 0.06 (38.3–38.4) | 38.43 | 33.20 | 153.2 |
| transformer/bgrl_seed1 | 36.31 ± 0.28 (35.7–36.7) | 38.35 ± 0.04 (38.3–38.4) | 38.46 | 33.26 | 153.8 |
| transformer/bgrl_seed2 | 36.23 ± 0.19 (36.0–36.7) | 38.41 ± 0.09 (38.3–38.5) | 38.59 | 33.00 | 153.9 |

Test accuracy (%) on arxiv-year. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
