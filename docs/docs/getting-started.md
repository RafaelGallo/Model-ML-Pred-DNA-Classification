# Getting started

## Requirements

Python 3.10 or newer.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

The source dataset ships with the repository at
`input/synthetic_dna_dataset.csv` - 3,000 synthetic DNA reads of 100 bases
each. No download step is required.

## Running the analysis

Execute both notebooks in order, which regenerates every figure, the trained
detector and the output reports:

```bash
make notebooks
```

Or open them interactively:

```bash
jupyter lab notebooks/
```

## Using the trained detector

`notebooks/05_dna_unsupervised_learning.ipynb` writes
`models/dna_anomaly_detector.joblib`. The Streamlit app loads it directly:

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
python -m pytest tests
```
