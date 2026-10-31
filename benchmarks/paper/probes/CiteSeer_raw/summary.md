| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 48.29 ± 0.43 (47.5–49.0) | 48.37 ± 0.90 (47.8–49.4) | 62.40 | 47.50 | 50.60 | 2185.4 |

Test accuracy (%) on CiteSeer. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
