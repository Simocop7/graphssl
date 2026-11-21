| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 65.40 | 72.90 | 64.60 | 459.4 |
| propagated A^1X | 69.90 | 75.90 | 67.60 | 428.5 |
| propagated A^2X | 75.30 | 79.90 | 72.80 | 395.4 |
| PCA-256 of A^2X | 54.40 | 79.80 | 71.20 | 215.2 |
| gcn/afgrl_seed0 | 79.40 | 78.50 | 73.80 | 158.4 |
| gcn/afgrl_seed1 | 77.80 | 78.90 | 69.10 | 157.5 |
| gcn/afgrl_seed2 | 78.90 | 78.70 | 71.90 | 155.9 |
| gcn/afgrl_seed3 | 79.70 | 77.70 | 69.00 | 157.3 |
| gcn/afgrl_seed4 | 79.70 | 78.80 | 70.90 | 158.7 |
| gcn/barlow_twins_seed0 | 81.20 | 81.00 | 74.70 | 152.7 |
| gcn/barlow_twins_seed1 | 81.30 | 82.20 | 74.60 | 153.4 |
| gcn/barlow_twins_seed2 | 81.30 | 81.00 | 75.30 | 152.7 |
| gcn/barlow_twins_seed3 | 79.20 | 79.20 | 74.80 | 152.4 |
| gcn/barlow_twins_seed4 | 79.80 | 79.60 | 74.50 | 152.7 |
| gcn/bgrl_seed0 | 80.40 | 79.40 | 72.20 | 150.5 |
| gcn/bgrl_seed1 | 80.50 | 79.90 | 73.10 | 150.6 |
| gcn/bgrl_seed2 | 80.10 | 79.40 | 69.50 | 151.1 |
| gcn/bgrl_seed3 | 81.60 | 80.20 | 71.90 | 150.9 |
| gcn/bgrl_seed4 | 78.40 | 78.60 | 71.30 | 148.3 |
| gcn/dgi_seed0 | 78.10 | 78.90 | 71.60 | 191.0 |
| gcn/dgi_seed1 | 81.10 | 81.10 | 74.20 | 189.5 |
| gcn/dgi_seed2 | 78.90 | 77.10 | 69.50 | 191.4 |
| gcn/dgi_seed3 | 75.50 | 77.50 | 71.00 | 189.1 |
| gcn/dgi_seed4 | 78.60 | 78.50 | 72.30 | 189.1 |
| gcn/graphcl_seed0 | 82.50 | 82.00 | 74.60 | 137.2 |
| gcn/graphcl_seed1 | 82.30 | 81.60 | 74.40 | 136.4 |
| gcn/graphcl_seed2 | 82.10 | 81.70 | 74.50 | 136.8 |
| gcn/graphcl_seed3 | 81.40 | 80.90 | 72.90 | 136.1 |
| gcn/graphcl_seed4 | 81.70 | 81.30 | 74.30 | 136.1 |
| gcn/graphdino_seed0 | 76.80 | 76.80 | 70.20 | 165.7 |
| gcn/graphdino_seed1 | 78.40 | 77.10 | 71.40 | 168.0 |
| gcn/graphdino_seed2 | 76.50 | 76.20 | 65.20 | 163.6 |
| gcn/graphdino_seed3 | 74.70 | 75.60 | 67.80 | 165.3 |
| gcn/graphdino_seed4 | 76.80 | 77.40 | 71.20 | 168.4 |
| gcn/graphmae_seed0 | 78.90 | 80.00 | 74.40 | 164.5 |
| gcn/graphmae_seed1 | 79.70 | 78.90 | 73.90 | 165.8 |
| gcn/graphmae_seed2 | 79.70 | 79.10 | 73.60 | 164.8 |
| gcn/graphmae_seed3 | 77.90 | 79.50 | 74.50 | 164.3 |
| gcn/graphmae_seed4 | 81.30 | 81.30 | 73.20 | 165.8 |
| gcn/vicreg_seed0 | 81.50 | 80.50 | 74.60 | 155.5 |
| gcn/vicreg_seed1 | 81.60 | 81.10 | 74.80 | 155.9 |
| gcn/vicreg_seed2 | 82.40 | 81.80 | 74.60 | 155.2 |
| gcn/vicreg_seed3 | 81.00 | 80.60 | 74.30 | 155.3 |
| gcn/vicreg_seed4 | 81.60 | 81.60 | 75.70 | 155.1 |
| gcn/bgrl_seed0 | 72.60 | 70.70 | 69.10 | 185.0 |
| gcn/bgrl_seed1 | 76.30 | 75.00 | 68.90 | 182.6 |
| gcn/bgrl_seed2 | 75.80 | 68.40 | 66.70 | 182.6 |
| gcn/bgrl_seed3 | 71.50 | 71.30 | 67.80 | 180.7 |
| gcn/bgrl_seed4 | 74.20 | 74.70 | 66.00 | 183.4 |

Test accuracy (%) on PubMed. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
