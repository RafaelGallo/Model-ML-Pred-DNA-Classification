"""Tests for the saved Isolation Forest anomaly detector."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from dna_classification.features import (
    QUALITY_FEATURE_NAMES,
    SEQUENCING_ADAPTER,
    extract_quality_features,
)


def score(bundle: dict, sequences: list[str]) -> np.ndarray:
    """Return anomaly scores, higher meaning more unusual."""
    features = extract_quality_features(sequences, adapter=bundle["adapter"])
    return -bundle["pipeline"].score_samples(features)


def test_bundle_contains_everything_the_app_needs(detector_bundle: dict):
    required = {
        "pipeline",
        "feature_names",
        "adapter",
        "score_thresholds",
        "model_name",
        "random_state",
        "training_reads",
        "holdout_metrics",
        "recall_by_artifact_at_5pct",
        "undetectable_artifacts",
    }
    assert required.issubset(detector_bundle)


def test_detector_is_an_isolation_forest_pipeline(detector_bundle: dict):
    pipeline = detector_bundle["pipeline"]
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["detector"], IsolationForest)
    assert detector_bundle["model_name"] == "Isolation Forest"


def test_feature_names_match_the_package_schema(detector_bundle: dict):
    assert detector_bundle["feature_names"] == list(QUALITY_FEATURE_NAMES)
    assert detector_bundle["adapter"] == SEQUENCING_ADAPTER


def test_thresholds_increase_as_the_budget_shrinks(detector_bundle: dict):
    """A tighter budget must demand a higher score to flag a read."""
    thresholds = detector_bundle["score_thresholds"]
    budgets = sorted(thresholds)
    values = [thresholds[budget] for budget in budgets]
    assert values == sorted(values, reverse=True)


def test_thresholds_flag_about_the_intended_share_of_clean_reads(
    dataset: pd.DataFrame, detector_bundle: dict
):
    scores = score(detector_bundle, dataset["Sequence"].tolist())
    for budget, threshold in detector_bundle["score_thresholds"].items():
        flagged = float((scores > threshold).mean())
        assert flagged == pytest.approx(budget, abs=0.01)


def test_artifacts_score_higher_than_clean_reads(detector_bundle: dict, clean_read):
    clean_score = score(detector_bundle, [clean_read])[0]
    homopolymer = score(detector_bundle, ["A" * 40 + clean_read[40:]])[0]
    adapter = score(
        detector_bundle,
        [clean_read[:40] + SEQUENCING_ADAPTER + clean_read[60:]],
    )[0]

    assert homopolymer > clean_score
    assert adapter > clean_score


def test_detectable_artifacts_clear_the_five_percent_threshold(
    detector_bundle: dict, clean_read
):
    threshold = detector_bundle["score_thresholds"][0.05]
    homopolymer = score(detector_bundle, ["A" * 40 + clean_read[40:]])[0]
    adapter = score(
        detector_bundle,
        [clean_read[:40] + SEQUENCING_ADAPTER + clean_read[60:]],
    )[0]

    assert homopolymer > threshold
    assert adapter > threshold


def test_reported_metrics_are_in_range(detector_bundle: dict):
    metrics = detector_bundle["holdout_metrics"]
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["recall_at_5pct"] <= 1.0
    assert metrics["recall_at_5pct"] <= metrics["recall_at_10pct"]


def test_chimeras_are_documented_as_undetectable(detector_bundle: dict):
    """The bundle must carry this limitation so the app can surface it."""
    assert "chimera" in detector_bundle["undetectable_artifacts"]
    assert detector_bundle["recall_by_artifact_at_5pct"]["chimera"] == 0.0


def test_scoring_is_deterministic(detector_bundle: dict, clean_read):
    first = score(detector_bundle, [clean_read, "A" * 100])
    second = score(detector_bundle, [clean_read, "A" * 100])
    assert np.allclose(first, second)
