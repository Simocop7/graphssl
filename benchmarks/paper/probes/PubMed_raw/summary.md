| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 69.29 ± 0.27 (68.8–69.7) | 68.80 ± 0.79 (68.2–69.7) | 72.90 | 67.00 | 64.60 | 456.3 |

Test accuracy (%) on PubMed. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
