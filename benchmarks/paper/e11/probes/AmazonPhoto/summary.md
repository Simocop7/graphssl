| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 86.76 | 87.75 | 58.59 | 678.0 |
| propagated A^1X | 92.70 | 92.92 | 91.99 | 473.6 |
| propagated A^2X | 92.92 | 92.79 | 91.98 | 313.9 |
| propagated A^3X | 92.21 | 92.75 | 91.45 | 223.8 |
| gcn/afgrl_seed0 | 92.30 | 92.42 | 90.28 | 108.3 |
| gcn/afgrl_seed1 | 92.47 | 92.50 | 90.82 | 101.0 |
| gcn/afgrl_seed2 | 92.61 | 92.57 | 90.21 | 106.6 |
| gcn/afgrl_seed3 | 91.99 | 92.06 | 90.33 | 104.7 |
| gcn/afgrl_seed4 | 91.81 | 92.03 | 90.54 | 107.6 |
| gcn/barlow_twins_seed0 | 91.47 | 90.05 | 91.49 | 66.9 |
| gcn/barlow_twins_seed1 | 92.12 | 90.11 | 91.63 | 68.0 |
| gcn/barlow_twins_seed2 | 92.16 | 90.85 | 91.47 | 66.1 |
| gcn/barlow_twins_seed3 | 92.14 | 89.72 | 91.58 | 68.1 |
| gcn/barlow_twins_seed4 | 92.39 | 90.52 | 91.55 | 70.0 |
| gcn/bgrl_seed0 | 92.60 | 92.04 | 91.83 | 255.7 |
| gcn/bgrl_seed1 | 92.66 | 92.12 | 91.90 | 254.6 |
| gcn/bgrl_seed2 | 92.75 | 92.01 | 91.65 | 256.9 |
| gcn/bgrl_seed3 | 92.39 | 91.99 | 91.63 | 255.1 |
| gcn/bgrl_seed4 | 92.70 | 92.24 | 91.75 | 255.5 |
| gcn/dgi_seed0 | 92.03 | 92.32 | 91.37 | 123.6 |
| gcn/dgi_seed1 | 91.83 | 91.41 | 91.23 | 126.2 |
| gcn/dgi_seed2 | 92.06 | 92.27 | 91.36 | 128.2 |
| gcn/dgi_seed3 | 92.29 | 92.04 | 90.80 | 126.2 |
| gcn/dgi_seed4 | 92.21 | 92.06 | 91.14 | 125.7 |
| gcn/graphcl_seed0 | 91.99 | 92.19 | 91.90 | 44.5 |
| gcn/graphcl_seed1 | 92.78 | 92.88 | 91.88 | 44.4 |
| gcn/graphcl_seed2 | 93.04 | 92.89 | 91.85 | 44.3 |
| gcn/graphcl_seed3 | 91.70 | 92.12 | 91.98 | 43.3 |
| gcn/graphcl_seed4 | 92.70 | 92.32 | 91.99 | 43.4 |
| gcn/graphdino_seed0 | 92.96 | 92.71 | 92.34 | 182.0 |
| gcn/graphdino_seed1 | 93.06 | 93.01 | 92.57 | 180.9 |
| gcn/graphdino_seed2 | 92.86 | 92.34 | 91.98 | 185.0 |
| gcn/graphdino_seed3 | 92.94 | 92.35 | 92.40 | 180.2 |
| gcn/graphdino_seed4 | 92.17 | 93.17 | 92.70 | 179.6 |
| gcn/graphmae_seed0 | 92.08 | 92.32 | 92.16 | 48.2 |
| gcn/graphmae_seed1 | 92.79 | 92.68 | 91.91 | 50.5 |
| gcn/graphmae_seed2 | 92.73 | 92.75 | 92.14 | 49.8 |
| gcn/graphmae_seed3 | 92.34 | 92.34 | 91.96 | 44.6 |
| gcn/graphmae_seed4 | 92.14 | 92.35 | 92.04 | 48.0 |
| gcn/vicreg_seed0 | 93.04 | 92.91 | 92.63 | 113.6 |
| gcn/vicreg_seed1 | 93.02 | 93.10 | 92.48 | 113.0 |
| gcn/vicreg_seed2 | 92.86 | 93.15 | 92.60 | 112.4 |
| gcn/vicreg_seed3 | 93.07 | 93.24 | 92.63 | 107.9 |
| gcn/vicreg_seed4 | 92.94 | 92.34 | 92.63 | 112.8 |

Test accuracy (%) on AmazonPhoto. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
