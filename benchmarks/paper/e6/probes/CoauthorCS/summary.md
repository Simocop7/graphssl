| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 92.28 ± 0.04 (92.2–92.3) | 92.34 ± 0.02 (92.3–92.4) | 92.13 | 91.31 | 88.61 | 4859.9 |
| propagated A^1X | 92.61 ± 0.03 (92.6–92.7) | 92.84 ± 0.03 (92.8–92.9) | 93.41 | 92.51 | 91.28 | 3867.6 |
| propagated A^2X | 92.18 ± 0.02 (92.1–92.2) | 92.47 ± 0.04 (92.4–92.5) | 93.07 | 92.32 | 91.22 | 2953.7 |
| PCA-256 of A^2X | 88.93 ± 0.25 (88.4–89.3) | 89.21 ± 0.08 (89.2–89.3) | 93.09 | 91.72 | 92.10 | 195.6 |

Test accuracy (%) on CoauthorCS. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
