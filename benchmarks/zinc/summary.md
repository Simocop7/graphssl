| Method | Frozen encoder | Fine-tuned | Untrained encoder | Rank | Epochs | Seeds |
|---|---|---|---|---|---|---|
| DGI | 0.932 ± 0.071 | 0.359 ± 0.018 | 0.593 ± 0.010 | 9.2 ± 3.3 | 100 | 3 |
| GraphCL | 0.571 ± 0.006 | 0.341 ± 0.007 | 0.593 ± 0.010 | 18.0 ± 0.2 | 100 | 3 |
| VICReg | 0.591 ± 0.010 | 0.335 ± 0.009 | 0.593 ± 0.010 | 18.5 ± 0.2 | 100 | 3 |
| Barlow Twins | 0.574 ± 0.005 | 0.354 ± 0.012 | 0.593 ± 0.010 | 15.7 ± 0.3 | 100 | 3 |
| BGRL | 0.700 ± 0.082 | 0.377 ± 0.007 | 0.593 ± 0.010 | 5.6 ± 4.9 | 100 | 3 |
| AFGRL | 0.671 ± 0.046 | 0.379 ± 0.007 | 0.593 ± 0.010 | 5.0 ± 1.3 | 100 | 3 |
| GraphDINO | 0.578 ± 0.009 | 0.361 ± 0.009 | 0.593 ± 0.010 | 15.6 ± 0.3 | 100 | 3 |
| *Supervised* † | 0.396 ± 0.028 | 0.364 ± 0.007 | 0.593 ± 0.010 | 7.5 ± 0.7 | 100 | 3 |

Test MAE, lower is better; mean ± sample std over the seeds. *Frozen encoder* and *Untrained encoder*: ridge regression on the graph embeddings. *Fine-tuned*: encoder and a linear head trained on the labels after pre-training, best-validation epoch.

† Not an SSL method: trained end to end on the labels from a random initialisation. Its *Fine-tuned* column is that training, the reference for the others.
