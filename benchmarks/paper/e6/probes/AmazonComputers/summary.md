| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 78.81 ± 0.33 (78.3–79.6) | 78.69 ± 0.33 (78.4–79.0) | 79.89 | 78.22 | 59.92 | 705.1 |
| propagated A^1X | 87.17 ± 0.39 (86.5–87.9) | 89.20 ± 0.24 (88.9–89.4) | 89.37 | 88.48 | 86.58 | 526.5 |
| propagated A^2X | 84.99 ± 0.50 (84.3–86.1) | 88.99 ± 0.16 (88.8–89.1) | 89.49 | 88.95 | 87.87 | 404.9 |
| PCA-256 of A^2X | 85.92 ± 0.13 (85.7–86.1) | 86.72 ± 0.31 (86.4–87.0) | 88.00 | 87.25 | 84.73 | 150.7 |

Test accuracy (%) on AmazonComputers. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
