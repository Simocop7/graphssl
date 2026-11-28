| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 91.31 | 92.13 | 88.61 | 4859.9 |
| propagated A^1X | 92.51 | 93.41 | 91.28 | 3867.6 |
| propagated A^2X | 92.32 | 93.07 | 91.22 | 2953.7 |
| propagated A^3X | 91.59 | 91.93 | 90.66 | 2281.3 |
| gcn/afgrl_seed0 | 93.15 | 93.28 | 92.45 | 243.7 |
| gcn/afgrl_seed1 | 93.19 | 93.34 | 92.51 | 253.6 |
| gcn/afgrl_seed2 | 93.20 | 93.02 | 92.36 | 248.8 |
| gcn/afgrl_seed3 | 93.14 | 93.20 | 92.32 | 250.9 |
| gcn/afgrl_seed4 | 93.42 | 93.37 | 92.45 | 256.1 |
| gcn/barlow_twins_seed0 | 92.71 | 93.07 | 92.36 | 97.2 |
| gcn/barlow_twins_seed1 | 93.12 | 93.21 | 92.39 | 97.0 |
| gcn/barlow_twins_seed2 | 93.33 | 93.22 | 92.49 | 97.3 |
| gcn/barlow_twins_seed3 | 92.85 | 93.21 | 92.58 | 95.4 |
| gcn/barlow_twins_seed4 | 92.54 | 93.09 | 92.21 | 96.3 |
| gcn/bgrl_seed0 | 93.52 | 93.59 | 93.05 | 135.8 |
| gcn/bgrl_seed1 | 93.46 | 93.38 | 92.88 | 132.1 |
| gcn/bgrl_seed2 | 93.28 | 93.55 | 92.83 | 132.4 |
| gcn/bgrl_seed3 | 93.43 | 93.63 | 92.74 | 132.0 |
| gcn/bgrl_seed4 | 93.49 | 93.63 | 92.92 | 132.9 |
| gcn/dgi_seed0 | 92.16 | 92.50 | 91.12 | 169.0 |
| gcn/dgi_seed1 | 91.77 | 92.08 | 90.85 | 167.0 |
| gcn/dgi_seed2 | 91.89 | 92.00 | 91.14 | 171.9 |
| gcn/dgi_seed3 | 91.95 | 92.23 | 91.21 | 170.6 |
| gcn/dgi_seed4 | 91.92 | 92.13 | 90.78 | 167.5 |
| gcn/graphcl_seed0 | 92.98 | 93.00 | 92.18 | 52.5 |
| gcn/graphcl_seed1 | 92.78 | 93.00 | 92.36 | 52.1 |
| gcn/graphcl_seed2 | 92.71 | 92.73 | 92.19 | 51.8 |
| gcn/graphcl_seed3 | 92.91 | 92.98 | 92.37 | 51.7 |
| gcn/graphcl_seed4 | 92.82 | 92.85 | 92.15 | 52.2 |
| gcn/graphdino_seed0 | 92.21 | 92.73 | 91.12 | 207.7 |
| gcn/graphdino_seed1 | 92.64 | 92.87 | 91.44 | 202.2 |
| gcn/graphdino_seed2 | 92.19 | 92.51 | 91.01 | 205.2 |
| gcn/graphdino_seed3 | 92.61 | 92.69 | 91.66 | 209.2 |
| gcn/graphdino_seed4 | 92.58 | 92.89 | 91.45 | 205.3 |
| gcn/graphmae_seed0 | 93.50 | 93.37 | 93.04 | 89.7 |
| gcn/graphmae_seed1 | 93.56 | 93.64 | 92.95 | 93.3 |
| gcn/graphmae_seed2 | 93.60 | 93.58 | 93.10 | 91.1 |
| gcn/graphmae_seed3 | 93.21 | 93.50 | 92.98 | 92.1 |
| gcn/graphmae_seed4 | 93.45 | 93.54 | 93.09 | 89.5 |
| gcn/vicreg_seed0 | 93.39 | 93.35 | 92.41 | 277.9 |
| gcn/vicreg_seed1 | 93.35 | 93.52 | 92.36 | 277.6 |
| gcn/vicreg_seed2 | 93.37 | 93.47 | 92.41 | 276.6 |
| gcn/vicreg_seed3 | 93.38 | 93.44 | 92.52 | 278.8 |
| gcn/vicreg_seed4 | 93.42 | 93.39 | 92.52 | 277.4 |

Test accuracy (%) on CoauthorCS. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
