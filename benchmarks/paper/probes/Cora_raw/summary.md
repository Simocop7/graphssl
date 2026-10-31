| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 45.34 ± 0.53 (44.3–46.3) | 46.97 ± 0.12 (46.9–47.1) | 60.20 | 52.60 | 46.30 | 1096.0 |

Test accuracy (%) on Cora. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
