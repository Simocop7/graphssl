| Embeddings | adam100_std | adam5000_std | graphssl_weakl2 | kNN | Rank |
|---|---|---|---|---|---|
| transformer/afgrl_seed0 | 39.02 ± 0.11 (38.8–39.2) | 40.19 ± 0.00 (40.2–40.2) | 40.29 | 37.71 | 59.6 |
| transformer/afgrl_seed1 | 38.68 ± 0.11 (38.4–38.8) | 40.24 ± 0.01 (40.2–40.3) | 40.16 | 37.73 | 61.5 |
| transformer/afgrl_seed2 | 38.75 ± 0.12 (38.5–39.0) | 40.30 ± 0.03 (40.3–40.3) | 40.38 | 38.10 | 54.0 |
| transformer/barlow_twins_seed0 | 37.12 ± 0.10 (36.9–37.3) | 37.78 ± 0.09 (37.7–37.9) | 37.79 | 29.77 | 107.1 |
| transformer/barlow_twins_seed1 | 37.10 ± 0.11 (36.9–37.3) | 37.94 ± 0.04 (37.9–38.0) | 37.99 | 29.94 | 108.8 |
| transformer/barlow_twins_seed2 | 37.06 ± 0.14 (36.8–37.3) | 37.66 ± 0.10 (37.6–37.7) | 37.64 | 29.77 | 108.0 |
| transformer/bgrl_seed0 | 37.00 ± 0.10 (36.9–37.2) | 37.61 ± 0.13 (37.5–37.7) | 37.73 | 29.69 | 148.5 |
| transformer/bgrl_seed1 | 37.06 ± 0.07 (37.0–37.2) | 37.53 ± 0.04 (37.5–37.6) | 37.56 | 30.05 | 145.8 |
| transformer/bgrl_seed2 | 37.02 ± 0.10 (36.8–37.3) | 37.56 ± 0.16 (37.4–37.7) | 37.68 | 29.55 | 146.6 |
| transformer/dgi_seed0 | 36.31 ± 0.24 (35.8–36.7) | 38.75 ± 0.09 (38.7–38.8) | 38.84 | 34.91 | 87.4 |
| transformer/dgi_seed1 | 36.65 ± 0.25 (36.2–37.1) | 39.23 ± 0.02 (39.2–39.3) | 39.32 | 35.16 | 85.1 |
| transformer/dgi_seed2 | 37.04 ± 0.27 (36.3–37.5) | 39.04 ± 0.02 (39.0–39.1) | 39.00 | 34.96 | 75.2 |
| transformer/graphcl_seed0 | 37.02 ± 0.12 (36.8–37.2) | 37.75 ± 0.04 (37.7–37.8) | 37.85 | 29.62 | 114.2 |
| transformer/graphcl_seed1 | 37.20 ± 0.11 (37.0–37.4) | 37.92 ± 0.05 (37.9–38.0) | 37.92 | 29.73 | 116.6 |
| transformer/graphcl_seed2 | 37.16 ± 0.08 (37.0–37.3) | 37.77 ± 0.02 (37.8–37.8) | 37.84 | 29.92 | 115.9 |
| transformer/graphdino_seed0 | 37.53 ± 0.09 (37.4–37.7) | 37.93 ± 0.16 (37.7–38.1) | 37.97 | 30.58 | 125.6 |
| transformer/graphdino_seed1 | 37.52 ± 0.10 (37.3–37.7) | 38.01 ± 0.08 (37.9–38.1) | 38.04 | 30.44 | 125.2 |
| transformer/graphdino_seed2 | 37.73 ± 0.08 (37.6–37.9) | 38.12 ± 0.05 (38.1–38.2) | 38.13 | 30.52 | 124.6 |
| transformer/vicreg_seed0 | 37.28 ± 0.10 (37.1–37.4) | 37.84 ± 0.02 (37.8–37.9) | 37.76 | 29.87 | 113.5 |
| transformer/vicreg_seed1 | 37.03 ± 0.10 (36.8–37.2) | 37.83 ± 0.05 (37.8–37.9) | 37.81 | 29.78 | 112.9 |
| transformer/vicreg_seed2 | 37.12 ± 0.12 (36.9–37.4) | 37.81 ± 0.03 (37.8–37.8) | 37.76 | 29.69 | 114.7 |
| transformer/bgrl_seed0 | 37.89 ± 0.09 (37.8–38.0) | 38.43 ± 0.04 (38.4–38.5) | 38.43 | 33.20 | 153.2 |
| transformer/bgrl_seed1 | 37.93 ± 0.10 (37.7–38.2) | 38.47 ± 0.03 (38.4–38.5) | 38.46 | 33.26 | 153.8 |
| transformer/bgrl_seed2 | 38.03 ± 0.09 (37.9–38.2) | 38.63 ± 0.08 (38.5–38.7) | 38.60 | 33.00 | 153.9 |

Test accuracy (%) on arxiv-year. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100_std`: adam100 on features standardized with the training rows' statistics.
- `adam5000_std`: adam5000 on features standardized with the training rows' statistics.
- `graphssl_weakl2`: the GraphSSL probe with the weakest L2 of its grid (1e-6), not selected.
