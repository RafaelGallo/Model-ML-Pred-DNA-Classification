"""Shared fixtures and import path setup for the test suite."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "src"
if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))

DATA_PATH = PROJECT_ROOT / "input" / "synthetic_dna_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "dna_classifier.joblib"
TUNED_MODEL_PATH = PROJECT_ROOT / "models" / "dna_classifier_tuned.joblib"
DETECTOR_PATH = PROJECT_ROOT / "models" / "dna_anomaly_detector.joblib"
PREDICTION_PATH = PROJECT_ROOT / "output" / "dna_predictions.csv"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """Load the source dataset, skipping the test when it is unavailable."""
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def model_bundle() -> dict:
    """Load the trained bundle, skipping when the notebooks have not been run."""
    if not MODEL_PATH.exists():
        pytest.skip("Run notebooks/02_dna_model_training.ipynb to create the model.")
    import joblib

    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def tuned_bundle() -> dict:
    """Load the tuned bundle, skipping when the tuning notebook has not run."""
    if not TUNED_MODEL_PATH.exists():
        pytest.skip(
            "Run notebooks/04_dna_hyperparameter_tuning.ipynb to create the "
            "tuned model."
        )
    import joblib

    return joblib.load(TUNED_MODEL_PATH)


@pytest.fixture(scope="session")
def saved_predictions() -> pd.DataFrame:
    """Load the saved held-out predictions CSV."""
    if not PREDICTION_PATH.exists():
        pytest.skip("Run notebooks/03_dna_prediction.ipynb to create the predictions.")
    return pd.read_csv(PREDICTION_PATH)


@pytest.fixture(scope="session")
def detector_bundle() -> dict:
    """Load the Isolation Forest bundle, skipping when it has not been built."""
    if not DETECTOR_PATH.exists():
        pytest.skip(
            "Run notebooks/05_dna_unsupervised_learning.ipynb to create "
            "models/dna_anomaly_detector.joblib."
        )
    import joblib

    return joblib.load(DETECTOR_PATH)


@pytest.fixture(scope="session")
def clean_read(dataset: pd.DataFrame) -> str:
    """Return a real read from the dataset.

    Synthetic strings like "ACGT" * 25 are a poor stand-in for a clean read:
    they are perfectly periodic, so they carry only four distinct 3-mers and an
    abnormally low k-mer entropy. Splicing an artifact into one can *raise* its
    entropy, which inverts the comparison a quality-control test is making.
    """
    return dataset["Sequence"].iat[0]
