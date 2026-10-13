| Method | Frozen encoder | Fine-tuned | Untrained encoder | Rank | Epochs | Seeds |
|---|---|---|---|---|---|---|
| DGI | 0.903 ± 0.182 | 0.363 ± 0.025 | 0.589 ± 0.011 | 11.9 ± 3.9 | 100 | 10 |
| GraphCL | 0.578 ± 0.006 | 0.343 ± 0.024 | 0.589 ± 0.011 | 18.0 ± 0.3 | 100 | 10 |
| VICReg | 0.585 ± 0.008 | 0.335 ± 0.027 | 0.589 ± 0.011 | 18.9 ± 0.4 | 100 | 10 |
| Barlow Twins | 0.585 ± 0.010 | 0.363 ± 0.010 | 0.589 ± 0.011 | 16.0 ± 0.4 | 100 | 10 |
| BGRL | 0.692 ± 0.060 | 0.369 ± 0.007 | 0.589 ± 0.011 | 4.0 ± 1.9 | 100 | 10 |
| AFGRL | 0.641 ± 0.039 | 0.382 ± 0.013 | 0.589 ± 0.011 | 8.9 ± 2.4 | 100 | 10 |
| GraphDINO | 0.579 ± 0.009 | 0.345 ± 0.017 | 0.589 ± 0.011 | 16.0 ± 0.7 | 100 | 10 |
| *Supervised* † | 0.402 ± 0.020 | 0.369 ± 0.005 | 0.589 ± 0.011 | 8.1 ± 0.8 | 100 | 10 |

Test MAE, lower is better; mean ± sample std over the seeds. *Frozen encoder* and *Untrained encoder*: ridge regression on the graph embeddings. *Fine-tuned*: encoder and a linear head trained on the labels after pre-training, best-validation epoch.

† Not an SSL method: trained end to end on the labels from a random initialisation. Its *Fine-tuned* column is that training, the reference for the others.
