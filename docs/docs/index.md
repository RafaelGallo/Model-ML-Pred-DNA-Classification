# DNA Sequence Analysis

Unsupervised machine learning on synthetic DNA sequence data: exploratory
analysis, dimensionality reduction with PCA and t-SNE, clustering with K-Means
and Agglomerative linkage, and an Isolation Forest anomaly detector for
sequencing quality control.

## What the project answers

1. **Do the sequences form natural groups?** No. Seven checks agree: the
   silhouette peaks at 0.03, the gap statistic never peaks, and K-Means and
   Agglomerative clustering agree with each other at ARI 0.13.
2. **Can unsupervised learning still add value?** Yes. An Isolation Forest
   flags contaminated reads at ROC AUC 0.872, recovering 65% of artifacts at a
   5% manual review budget and 75% at 10%.

## Contents

- `notebooks/01_dna_data_analysis.ipynb` - exploratory data analysis.
- `notebooks/05_dna_unsupervised_learning.ipynb` - PCA, t-SNE, K-Means,
  Agglomerative clustering, cluster validation, and anomaly detection.
- `src/dna_classification/features.py` - shared feature engineering.
- `app/streamlit_app.py` - quality-control screening interface.

## Commands

The Makefile holds the entry points for common tasks:

- `make requirements` - install dependencies.
- `make notebooks` - execute both notebooks and refresh every output.
- `make app` - launch the Streamlit quality-control app.
- `make test` - run the pytest suite.
- `make lint` / `make format` - check or apply code style.

See the project [README](https://github.com/RafaelGallo/Model-ML-Pred-DNA-Classification)
for the full write-up with figures.
