| Embeddings | adam100_std | adam5000_std | graphssl_weakl2 | kNN | Rank |
|---|---|---|---|---|---|
| transformer/afgrl_seed0 | 67.05 ± 0.09 (66.9–67.2) | 68.93 ± 0.04 (68.9–69.0) | 68.84 | 66.57 | 59.6 |
| transformer/afgrl_seed1 | 67.15 ± 0.14 (66.9–67.4) | 69.08 ± 0.04 (69.1–69.1) | 69.00 | 66.40 | 61.5 |
| transformer/afgrl_seed2 | 67.01 ± 0.11 (66.8–67.3) | 69.13 ± 0.06 (69.1–69.2) | 69.09 | 66.58 | 54.0 |
| transformer/barlow_twins_seed0 | 57.37 ± 0.17 (57.1–57.7) | 63.08 ± 0.08 (63.0–63.2) | 62.85 | 46.95 | 107.1 |
| transformer/barlow_twins_seed1 | 57.26 ± 0.14 (57.0–57.5) | 63.09 ± 0.02 (63.1–63.1) | 63.13 | 47.24 | 108.8 |
| transformer/barlow_twins_seed2 | 57.24 ± 0.13 (57.0–57.5) | 62.93 ± 0.12 (62.8–63.0) | 62.80 | 46.47 | 108.0 |
| transformer/bgrl_seed0 | 58.49 ± 0.30 (57.6–58.9) | 61.81 ± 0.16 (61.7–62.0) | 61.69 | 46.95 | 148.5 |
| transformer/bgrl_seed1 | 58.69 ± 0.16 (58.3–59.0) | 61.98 ± 0.04 (62.0–62.0) | 61.85 | 47.27 | 145.8 |
| transformer/bgrl_seed2 | 58.17 ± 0.20 (57.5–58.5) | 61.84 ± 0.03 (61.8–61.9) | 61.69 | 46.79 | 146.6 |
| transformer/dgi_seed0 | 58.83 ± 0.63 (57.3–59.9) | 65.15 ± 0.04 (65.1–65.2) | 64.83 | 53.24 | 87.4 |
| transformer/dgi_seed1 | 59.25 ± 0.55 (58.0–60.0) | 65.22 ± 0.04 (65.2–65.3) | 65.10 | 55.36 | 85.1 |
| transformer/dgi_seed2 | 57.73 ± 0.49 (56.9–58.9) | 65.02 ± 0.02 (65.0–65.0) | 64.90 | 54.00 | 75.2 |
| transformer/graphcl_seed0 | 57.22 ± 0.13 (57.0–57.4) | 62.83 ± 0.03 (62.8–62.9) | 62.93 | 47.05 | 114.2 |
| transformer/graphcl_seed1 | 57.32 ± 0.11 (57.1–57.5) | 62.55 ± 0.11 (62.5–62.7) | 62.54 | 47.34 | 116.6 |
| transformer/graphcl_seed2 | 57.32 ± 0.14 (57.0–57.6) | 62.95 ± 0.09 (62.9–63.0) | 62.95 | 46.73 | 115.9 |
| transformer/graphdino_seed0 | 60.65 ± 0.10 (60.5–60.9) | 63.11 ± 0.06 (63.0–63.2) | 63.03 | 49.13 | 125.6 |
| transformer/graphdino_seed1 | 60.58 ± 0.13 (60.4–60.9) | 63.39 ± 0.03 (63.4–63.4) | 63.33 | 49.20 | 125.2 |
| transformer/graphdino_seed2 | 61.05 ± 0.09 (60.9–61.3) | 63.40 ± 0.06 (63.4–63.5) | 63.13 | 49.31 | 124.6 |
| transformer/vicreg_seed0 | 57.08 ± 0.12 (56.8–57.3) | 62.76 ± 0.04 (62.7–62.8) | 62.66 | 46.57 | 113.5 |
| transformer/vicreg_seed1 | 57.08 ± 0.11 (56.9–57.3) | 62.54 ± 0.05 (62.5–62.6) | 62.51 | 47.03 | 112.9 |
| transformer/vicreg_seed2 | 56.83 ± 0.10 (56.7–57.1) | 62.09 ± 0.09 (62.0–62.2) | 62.16 | 46.43 | 114.7 |
| transformer/bgrl_seed0 | 62.94 ± 0.10 (62.7–63.1) | 66.63 ± 0.01 (66.6–66.6) | 66.60 | 52.47 | 153.2 |
| transformer/bgrl_seed1 | 63.01 ± 0.10 (62.8–63.1) | 66.47 ± 0.08 (66.4–66.5) | 66.30 | 52.35 | 153.8 |
| transformer/bgrl_seed2 | 63.16 ± 0.12 (62.9–63.5) | 66.42 ± 0.05 (66.4–66.5) | 66.45 | 52.53 | 153.9 |

Test accuracy (%) on ogbn-arxiv. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100_std`: adam100 on features standardized with the training rows' statistics.
- `adam5000_std`: adam5000 on features standardized with the training rows' statistics.
- `graphssl_weakl2`: the GraphSSL probe with the weakest L2 of its grid (1e-6), not selected.
