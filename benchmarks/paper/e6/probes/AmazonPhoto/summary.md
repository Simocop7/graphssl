| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 87.00 ± 0.37 (86.4–87.5) | 87.38 ± 0.21 (87.2–87.6) | 87.75 | 86.76 | 58.59 | 678.0 |
| propagated A^1X | 92.09 ± 0.16 (91.8–92.4) | 92.85 ± 0.15 (92.7–93.0) | 92.92 | 92.70 | 91.99 | 473.6 |
| propagated A^2X | 92.03 ± 0.12 (91.8–92.3) | 92.69 ± 0.22 (92.5–92.9) | 92.79 | 92.92 | 91.98 | 313.9 |
| PCA-256 of A^2X | 89.05 ± 0.18 (88.7–89.3) | 89.32 ± 0.26 (89.2–89.6) | 92.52 | 89.48 | 89.92 | 140.1 |

Test accuracy (%) on AmazonPhoto. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
