| Embeddings | adam100 | adam5000 | graphssl | kNN | Rank |
|---|---|---|---|---|---|
| raw features | 34.61 ± 0.12 (34.4–34.8) | 34.94 ± 0.06 (34.9–35.0) | 34.89 | 31.41 | 112.6 |
| propagated A^1X | 38.57 ± 0.20 (38.1–39.0) | 39.83 ± 0.02 (39.8–39.8) | 39.85 | 34.24 | 82.0 |
| propagated A^2X | 37.47 ± 0.19 (37.1–37.8) | 38.97 ± 0.02 (38.9–39.0) | 38.99 | 40.02 | 79.6 |

Test accuracy (%) on arxiv-year. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
