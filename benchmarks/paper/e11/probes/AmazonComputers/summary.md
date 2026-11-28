| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 78.22 | 79.89 | 59.92 | 705.1 |
| propagated A^1X | 88.48 | 89.37 | 86.58 | 526.5 |
| propagated A^2X | 88.95 | 89.49 | 87.87 | 404.9 |
| propagated A^3X | 88.48 | 88.75 | 87.31 | 334.2 |
| gcn/afgrl_seed0 | 89.02 | 89.24 | 86.04 | 225.5 |
| gcn/afgrl_seed1 | 89.02 | 88.94 | 86.54 | 231.5 |
| gcn/afgrl_seed2 | 88.82 | 88.43 | 86.53 | 219.7 |
| gcn/afgrl_seed3 | 89.00 | 89.04 | 86.30 | 228.4 |
| gcn/afgrl_seed4 | 89.14 | 89.54 | 86.68 | 224.9 |
| gcn/barlow_twins_seed0 | 88.37 | 87.85 | 85.23 | 131.9 |
| gcn/barlow_twins_seed1 | 88.54 | 88.40 | 85.08 | 129.4 |
| gcn/barlow_twins_seed2 | 88.13 | 88.14 | 85.04 | 131.4 |
| gcn/barlow_twins_seed3 | 87.27 | 88.72 | 85.44 | 132.5 |
| gcn/barlow_twins_seed4 | 88.36 | 88.27 | 85.66 | 133.6 |
| gcn/bgrl_seed0 | 89.01 | 88.68 | 87.48 | 50.7 |
| gcn/bgrl_seed1 | 88.92 | 88.53 | 87.89 | 51.3 |
| gcn/bgrl_seed2 | 88.85 | 88.48 | 87.62 | 51.3 |
| gcn/bgrl_seed3 | 88.81 | 88.45 | 87.66 | 51.5 |
| gcn/bgrl_seed4 | 89.24 | 89.24 | 88.01 | 51.2 |
| gcn/dgi_seed0 | 87.99 | 88.31 | 86.84 | 201.6 |
| gcn/dgi_seed1 | 87.81 | 87.87 | 86.32 | 191.5 |
| gcn/dgi_seed2 | 87.97 | 87.45 | 86.52 | 210.8 |
| gcn/dgi_seed3 | 87.59 | 88.22 | 86.12 | 210.1 |
| gcn/dgi_seed4 | 88.30 | 88.35 | 86.34 | 179.5 |
| gcn/graphcl_seed0 | 89.00 | 89.44 | 86.56 | 144.8 |
| gcn/graphcl_seed1 | 88.87 | 89.22 | 86.97 | 144.0 |
| gcn/graphcl_seed2 | 89.51 | 89.70 | 86.68 | 141.7 |
| gcn/graphcl_seed3 | 89.44 | 89.55 | 87.09 | 146.8 |
| gcn/graphcl_seed4 | 89.47 | 89.51 | 87.19 | 146.7 |
| gcn/graphdino_seed0 | 89.27 | 89.27 | 87.84 | 216.4 |
| gcn/graphdino_seed1 | 89.01 | 89.08 | 88.14 | 222.4 |
| gcn/graphdino_seed2 | 89.08 | 89.23 | 87.86 | 220.5 |
| gcn/graphdino_seed3 | 89.43 | 89.31 | 87.99 | 219.3 |
| gcn/graphdino_seed4 | 89.03 | 88.86 | 87.78 | 217.4 |
| gcn/graphmae_seed0 | 89.06 | 89.14 | 88.95 | 61.3 |
| gcn/graphmae_seed1 | 89.08 | 89.43 | 88.83 | 64.0 |
| gcn/graphmae_seed2 | 89.22 | 89.82 | 88.91 | 60.9 |
| gcn/graphmae_seed3 | 89.46 | 89.44 | 88.95 | 64.0 |
| gcn/graphmae_seed4 | 89.80 | 89.38 | 89.28 | 62.7 |
| gcn/vicreg_seed0 | 89.29 | 89.04 | 88.32 | 126.4 |
| gcn/vicreg_seed1 | 88.89 | 89.08 | 88.18 | 125.1 |
| gcn/vicreg_seed2 | 89.33 | 88.96 | 88.33 | 126.7 |
| gcn/vicreg_seed3 | 89.28 | 89.30 | 88.16 | 124.5 |
| gcn/vicreg_seed4 | 89.30 | 89.31 | 88.40 | 125.3 |

Test accuracy (%) on AmazonComputers. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
