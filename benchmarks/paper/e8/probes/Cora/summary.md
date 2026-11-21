| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 46.90 | 60.20 | 46.60 | 1057.7 |
| propagated A^1X | 68.50 | 78.20 | 64.60 | 884.7 |
| propagated A^2X | 69.70 | 80.40 | 69.90 | 713.4 |
| PCA-256 of A^2X | 60.60 | 80.70 | 74.60 | 227.1 |
| gcn/afgrl_seed0 | 76.00 | 76.20 | 73.20 | 178.7 |
| gcn/afgrl_seed1 | 80.00 | 78.00 | 71.90 | 177.9 |
| gcn/afgrl_seed2 | 77.50 | 77.40 | 67.70 | 179.4 |
| gcn/afgrl_seed3 | 76.90 | 78.30 | 72.50 | 179.0 |
| gcn/afgrl_seed4 | 78.70 | 79.60 | 71.70 | 178.6 |
| gcn/barlow_twins_seed0 | 79.30 | 80.10 | 74.90 | 162.3 |
| gcn/barlow_twins_seed1 | 82.00 | 80.80 | 76.10 | 163.9 |
| gcn/barlow_twins_seed2 | 79.90 | 81.00 | 75.60 | 163.5 |
| gcn/barlow_twins_seed3 | 81.20 | 82.00 | 77.20 | 163.0 |
| gcn/barlow_twins_seed4 | 80.80 | 81.10 | 74.80 | 163.7 |
| gcn/bgrl_seed0 | 80.90 | 79.60 | 73.80 | 170.7 |
| gcn/bgrl_seed1 | 81.10 | 81.00 | 75.50 | 170.3 |
| gcn/bgrl_seed2 | 81.90 | 82.30 | 71.90 | 170.0 |
| gcn/bgrl_seed3 | 77.30 | 78.30 | 72.20 | 170.7 |
| gcn/bgrl_seed4 | 78.90 | 82.70 | 75.20 | 169.9 |
| gcn/dgi_seed0 | 78.30 | 77.20 | 72.30 | 198.1 |
| gcn/dgi_seed1 | 77.20 | 76.70 | 73.20 | 199.3 |
| gcn/dgi_seed2 | 75.70 | 76.80 | 68.10 | 197.0 |
| gcn/dgi_seed3 | 77.10 | 77.00 | 71.50 | 198.2 |
| gcn/dgi_seed4 | 76.90 | 77.00 | 72.80 | 198.2 |
| gcn/graphcl_seed0 | 81.50 | 80.30 | 76.50 | 162.5 |
| gcn/graphcl_seed1 | 83.70 | 82.90 | 74.90 | 162.0 |
| gcn/graphcl_seed2 | 82.20 | 82.20 | 72.00 | 163.0 |
| gcn/graphcl_seed3 | 82.00 | 82.20 | 76.30 | 162.6 |
| gcn/graphcl_seed4 | 82.10 | 81.60 | 77.40 | 162.2 |
| gcn/graphdino_seed0 | 76.20 | 76.30 | 64.20 | 179.6 |
| gcn/graphdino_seed1 | 75.80 | 75.60 | 63.50 | 181.1 |
| gcn/graphdino_seed2 | 73.30 | 73.60 | 62.20 | 178.3 |
| gcn/graphdino_seed3 | 76.40 | 77.50 | 68.80 | 179.9 |
| gcn/graphdino_seed4 | 75.70 | 74.70 | 62.80 | 181.5 |
| gcn/graphmae_seed0 | 80.80 | 80.40 | 73.20 | 187.5 |
| gcn/graphmae_seed1 | 80.60 | 81.50 | 72.80 | 187.2 |
| gcn/graphmae_seed2 | 78.80 | 79.20 | 74.20 | 187.6 |
| gcn/graphmae_seed3 | 80.60 | 81.00 | 74.40 | 188.3 |
| gcn/graphmae_seed4 | 78.70 | 80.70 | 74.20 | 187.5 |
| gcn/vicreg_seed0 | 78.60 | 80.00 | 72.00 | 163.4 |
| gcn/vicreg_seed1 | 80.50 | 81.10 | 73.10 | 165.9 |
| gcn/vicreg_seed2 | 81.10 | 80.70 | 70.30 | 164.4 |
| gcn/vicreg_seed3 | 81.00 | 80.20 | 74.80 | 164.6 |
| gcn/vicreg_seed4 | 79.50 | 80.30 | 73.40 | 164.4 |
| gcn/bgrl_seed0 | 71.70 | 69.60 | 59.20 | 187.2 |
| gcn/bgrl_seed1 | 73.10 | 71.00 | 62.10 | 188.3 |
| gcn/bgrl_seed2 | 70.30 | 72.20 | 58.50 | 186.8 |
| gcn/bgrl_seed3 | 70.80 | 70.90 | 61.10 | 188.2 |
| gcn/bgrl_seed4 | 72.40 | 73.60 | 56.60 | 189.4 |

Test accuracy (%) on Cora. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
