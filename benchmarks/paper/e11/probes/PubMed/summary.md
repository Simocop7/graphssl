| Embeddings | graphssl | liblinear | kNN | Rank |
|---|---|---|---|---|
| raw features | 67.00 | 72.90 | 64.60 | 456.3 |
| propagated A^1X | 69.90 | 76.10 | 67.20 | 424.9 |
| propagated A^2X | 75.10 | 79.70 | 72.50 | 391.1 |
| propagated A^3X | 74.90 | 78.20 | 73.00 | 359.6 |
| gcn/afgrl_seed0 | 78.90 | 79.10 | 71.00 | 330.2 |
| gcn/afgrl_seed1 | 78.40 | 79.40 | 69.80 | 330.0 |
| gcn/afgrl_seed2 | 79.40 | 79.40 | 71.80 | 329.8 |
| gcn/afgrl_seed3 | 78.60 | 78.20 | 71.30 | 331.4 |
| gcn/afgrl_seed4 | 78.70 | 79.30 | 70.60 | 329.4 |
| gcn/barlow_twins_seed0 | 79.30 | 82.40 | 77.50 | 206.1 |
| gcn/barlow_twins_seed1 | 81.70 | 82.70 | 75.20 | 206.9 |
| gcn/barlow_twins_seed2 | 80.90 | 81.10 | 73.30 | 136.0 |
| gcn/barlow_twins_seed3 | 80.60 | 81.80 | 74.10 | 139.7 |
| gcn/barlow_twins_seed4 | 79.20 | 81.90 | 72.60 | 133.8 |
| gcn/bgrl_seed0 | 83.10 | 81.10 | 74.60 | 73.1 |
| gcn/bgrl_seed1 | 81.30 | 81.20 | 75.10 | 73.6 |
| gcn/bgrl_seed2 | 81.60 | 81.70 | 77.60 | 73.1 |
| gcn/bgrl_seed3 | 80.60 | 81.60 | 75.70 | 71.9 |
| gcn/bgrl_seed4 | 81.60 | 81.90 | 75.00 | 71.9 |
| gcn/dgi_seed0 | 81.40 | 79.10 | 73.30 | 181.4 |
| gcn/dgi_seed1 | 80.20 | 79.60 | 71.00 | 179.1 |
| gcn/dgi_seed2 | 79.20 | 79.90 | 72.40 | 184.0 |
| gcn/dgi_seed3 | 80.50 | 79.10 | 71.10 | 182.6 |
| gcn/dgi_seed4 | 81.20 | 80.00 | 73.00 | 185.7 |
| gcn/graphcl_seed0 | 80.60 | 80.90 | 74.30 | 62.5 |
| gcn/graphcl_seed1 | 81.00 | 82.10 | 74.90 | 62.5 |
| gcn/graphcl_seed2 | 81.90 | 81.60 | 73.50 | 62.6 |
| gcn/graphcl_seed3 | 81.10 | 81.60 | 74.90 | 62.3 |
| gcn/graphcl_seed4 | 79.50 | 80.40 | 73.60 | 62.3 |
| gcn/graphdino_seed0 | 81.80 | 82.20 | 75.60 | 54.0 |
| gcn/graphdino_seed1 | 81.60 | 81.60 | 76.00 | 53.3 |
| gcn/graphdino_seed2 | 83.20 | 83.10 | 78.50 | 56.6 |
| gcn/graphdino_seed3 | 82.00 | 83.30 | 75.80 | 52.7 |
| gcn/graphdino_seed4 | 81.20 | 81.60 | 73.10 | 56.5 |
| gcn/graphmae_seed0 | 81.40 | 81.50 | 73.60 | 309.1 |
| gcn/graphmae_seed1 | 79.50 | 79.60 | 72.90 | 309.7 |
| gcn/graphmae_seed2 | 79.60 | 78.70 | 72.50 | 307.5 |
| gcn/graphmae_seed3 | 79.10 | 80.00 | 72.80 | 309.2 |
| gcn/graphmae_seed4 | 80.20 | 80.40 | 72.90 | 307.3 |
| gcn/vicreg_seed0 | 81.00 | 82.70 | 74.40 | 96.4 |
| gcn/vicreg_seed1 | 82.40 | 82.30 | 75.40 | 97.6 |
| gcn/vicreg_seed2 | 82.00 | 82.10 | 74.90 | 95.2 |
| gcn/vicreg_seed3 | 81.80 | 83.10 | 74.70 | 91.9 |
| gcn/vicreg_seed4 | 82.00 | 83.30 | 75.10 | 92.1 |

Test accuracy (%) on PubMed. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
