| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 47.50 | 62.40 | 50.60 | 2185.4 |
| propagated A^1X | 55.50 | 71.60 | 62.40 | 1572.3 |
| propagated A^2X | 46.80 | 72.30 | 65.50 | 1226.8 |
| propagated A^3X | 47.60 | 72.50 | 65.00 | 1025.0 |
| gcn/afgrl_seed0 | 70.00 | 68.90 | 65.40 | 209.2 |
| gcn/afgrl_seed1 | 70.70 | 72.40 | 65.30 | 210.3 |
| gcn/afgrl_seed2 | 69.10 | 70.60 | 67.00 | 209.3 |
| gcn/afgrl_seed3 | 70.30 | 70.60 | 62.60 | 208.3 |
| gcn/afgrl_seed4 | 69.90 | 71.00 | 66.10 | 209.9 |
| gcn/barlow_twins_seed0 | 71.10 | 70.40 | 65.70 | 110.1 |
| gcn/barlow_twins_seed1 | 68.40 | 67.60 | 65.00 | 110.2 |
| gcn/barlow_twins_seed2 | 67.40 | 66.90 | 61.00 | 110.5 |
| gcn/barlow_twins_seed3 | 68.60 | 70.70 | 63.60 | 110.2 |
| gcn/barlow_twins_seed4 | 69.20 | 71.20 | 64.70 | 110.0 |
| gcn/bgrl_seed0 | 69.30 | 69.70 | 63.10 | 213.2 |
| gcn/bgrl_seed1 | 67.80 | 69.00 | 59.80 | 213.3 |
| gcn/bgrl_seed2 | 67.60 | 68.20 | 60.20 | 212.8 |
| gcn/bgrl_seed3 | 68.70 | 69.90 | 61.30 | 212.4 |
| gcn/bgrl_seed4 | 66.70 | 66.90 | 59.10 | 213.4 |
| gcn/dgi_seed0 | 66.80 | 68.80 | 65.20 | 263.2 |
| gcn/dgi_seed1 | 68.40 | 69.50 | 67.80 | 286.1 |
| gcn/dgi_seed2 | 68.60 | 68.90 | 65.50 | 269.7 |
| gcn/dgi_seed3 | 68.50 | 69.60 | 65.10 | 266.3 |
| gcn/dgi_seed4 | 72.00 | 71.60 | 67.80 | 276.6 |
| gcn/graphcl_seed0 | 70.20 | 70.30 | 65.00 | 218.1 |
| gcn/graphcl_seed1 | 70.30 | 72.90 | 65.20 | 217.8 |
| gcn/graphcl_seed2 | 71.70 | 71.80 | 64.90 | 217.1 |
| gcn/graphcl_seed3 | 72.20 | 71.00 | 65.50 | 217.2 |
| gcn/graphcl_seed4 | 69.90 | 70.20 | 63.30 | 217.5 |
| gcn/graphdino_seed0 | 68.80 | 70.10 | 67.60 | 301.3 |
| gcn/graphdino_seed1 | 72.00 | 68.70 | 67.20 | 304.0 |
| gcn/graphdino_seed2 | 69.70 | 70.30 | 65.10 | 302.8 |
| gcn/graphdino_seed3 | 70.20 | 70.20 | 66.70 | 306.0 |
| gcn/graphdino_seed4 | 69.30 | 69.10 | 67.30 | 307.1 |
| gcn/graphmae_seed0 | 72.10 | 70.80 | 68.00 | 65.8 |
| gcn/graphmae_seed1 | 72.10 | 73.20 | 69.10 | 64.8 |
| gcn/graphmae_seed2 | 71.60 | 70.80 | 69.30 | 65.0 |
| gcn/graphmae_seed3 | 70.60 | 71.80 | 68.10 | 66.6 |
| gcn/graphmae_seed4 | 71.40 | 71.00 | 67.70 | 63.1 |
| gcn/vicreg_seed0 | 71.30 | 71.30 | 68.20 | 161.9 |
| gcn/vicreg_seed1 | 70.60 | 70.40 | 66.50 | 159.0 |
| gcn/vicreg_seed2 | 71.20 | 70.90 | 69.00 | 158.9 |
| gcn/vicreg_seed3 | 70.10 | 69.60 | 67.70 | 159.6 |
| gcn/vicreg_seed4 | 71.10 | 71.50 | 66.00 | 159.6 |

Test accuracy (%) on CiteSeer. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
