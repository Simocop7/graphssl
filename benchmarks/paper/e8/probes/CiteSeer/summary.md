| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 47.20 | 62.40 | 50.50 | 2183.6 |
| propagated A^1X | 54.90 | 71.20 | 63.50 | 1572.7 |
| propagated A^2X | 46.40 | 71.60 | 64.60 | 1229.5 |
| PCA-256 of A^2X | 47.40 | 72.00 | 66.30 | 243.0 |
| gcn/afgrl_seed0 | 61.70 | 60.10 | 56.50 | 198.0 |
| gcn/afgrl_seed1 | 61.70 | 62.80 | 54.10 | 198.3 |
| gcn/afgrl_seed2 | 65.00 | 65.10 | 56.60 | 198.6 |
| gcn/afgrl_seed3 | 62.00 | 63.60 | 56.60 | 198.5 |
| gcn/afgrl_seed4 | 61.80 | 62.60 | 57.20 | 198.2 |
| gcn/barlow_twins_seed0 | 69.20 | 67.70 | 62.80 | 183.8 |
| gcn/barlow_twins_seed1 | 71.00 | 71.60 | 66.60 | 183.1 |
| gcn/barlow_twins_seed2 | 67.20 | 67.80 | 64.60 | 182.8 |
| gcn/barlow_twins_seed3 | 70.30 | 68.60 | 64.00 | 182.8 |
| gcn/barlow_twins_seed4 | 69.70 | 68.20 | 61.30 | 183.2 |
| gcn/bgrl_seed0 | 66.10 | 67.10 | 60.30 | 191.0 |
| gcn/bgrl_seed1 | 68.40 | 68.80 | 63.90 | 191.2 |
| gcn/bgrl_seed2 | 64.80 | 68.00 | 56.90 | 191.0 |
| gcn/bgrl_seed3 | 66.90 | 66.50 | 61.20 | 190.9 |
| gcn/bgrl_seed4 | 68.10 | 67.90 | 61.20 | 191.0 |
| gcn/dgi_seed0 | 65.90 | 67.00 | 59.50 | 210.6 |
| gcn/dgi_seed1 | 66.10 | 67.10 | 58.60 | 211.4 |
| gcn/dgi_seed2 | 64.90 | 66.60 | 57.50 | 211.9 |
| gcn/dgi_seed3 | 67.90 | 68.30 | 62.50 | 208.5 |
| gcn/dgi_seed4 | 65.10 | 64.40 | 56.60 | 212.0 |
| gcn/graphcl_seed0 | 70.40 | 71.50 | 62.00 | 180.8 |
| gcn/graphcl_seed1 | 70.10 | 71.20 | 64.10 | 180.1 |
| gcn/graphcl_seed2 | 67.90 | 68.70 | 62.90 | 180.2 |
| gcn/graphcl_seed3 | 71.50 | 72.20 | 65.90 | 179.9 |
| gcn/graphcl_seed4 | 69.10 | 68.60 | 62.80 | 180.3 |
| gcn/graphdino_seed0 | 59.40 | 59.60 | 49.20 | 179.4 |
| gcn/graphdino_seed1 | 59.70 | 59.60 | 45.20 | 177.5 |
| gcn/graphdino_seed2 | 59.20 | 60.80 | 47.80 | 180.6 |
| gcn/graphdino_seed3 | 58.90 | 60.90 | 47.10 | 178.4 |
| gcn/graphdino_seed4 | 56.80 | 57.70 | 47.20 | 182.2 |
| gcn/graphmae_seed0 | 70.50 | 71.30 | 64.20 | 204.3 |
| gcn/graphmae_seed1 | 68.70 | 69.90 | 62.80 | 204.3 |
| gcn/graphmae_seed2 | 69.60 | 70.30 | 62.50 | 205.1 |
| gcn/graphmae_seed3 | 68.40 | 69.50 | 62.20 | 205.2 |
| gcn/graphmae_seed4 | 69.10 | 69.80 | 63.50 | 204.9 |
| gcn/vicreg_seed0 | 65.60 | 66.60 | 63.10 | 183.0 |
| gcn/vicreg_seed1 | 68.60 | 68.50 | 63.40 | 181.7 |
| gcn/vicreg_seed2 | 68.10 | 67.80 | 62.20 | 180.2 |
| gcn/vicreg_seed3 | 68.10 | 67.50 | 62.70 | 182.9 |
| gcn/vicreg_seed4 | 68.40 | 66.50 | 62.50 | 183.2 |
| gcn/bgrl_seed0 | 59.20 | 59.00 | 48.70 | 205.7 |
| gcn/bgrl_seed1 | 59.40 | 60.50 | 47.80 | 204.7 |
| gcn/bgrl_seed2 | 55.90 | 60.40 | 52.60 | 204.7 |
| gcn/bgrl_seed3 | 56.10 | 56.80 | 49.30 | 204.9 |
| gcn/bgrl_seed4 | 56.00 | 58.10 | 49.90 | 206.1 |

Test accuracy (%) on CiteSeer. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
