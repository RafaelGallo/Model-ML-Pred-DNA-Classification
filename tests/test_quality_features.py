"""Tests for the quality-control features used by the anomaly detector."""

import numpy as np
import pandas as pd
import pytest

from dna_classification.features import (
    QUALITY_FEATURE_NAMES,
    SEQUENCING_ADAPTER,
    extract_quality_features,
    longest_homopolymer_run,
)

CLEAN = "ACGT" * 25


def test_schema_matches_the_declared_names():
    features = extract_quality_features([CLEAN])
    assert list(features.columns) == list(QUALITY_FEATURE_NAMES)
    assert len(features) == 1


def test_gc_content_is_a_percentage():
    assert extract_quality_features(["ACGT" * 25]).gc_content.iat[0] == 50.0
    assert extract_quality_features(["GCGC" * 25]).gc_content.iat[0] == 100.0
    assert extract_quality_features(["ATAT" * 25]).gc_content.iat[0] == 0.0


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [("ACGT", 1), ("AACGT", 2), ("A" * 30 + "CGT" * 20, 30), ("ACGTTTTT", 5)],
)
def test_longest_homopolymer_run(sequence, expected):
    assert longest_homopolymer_run(sequence) == expected


def test_longest_homopolymer_run_validates_its_input():
    with pytest.raises(ValueError, match="Unsupported DNA symbols"):
        longest_homopolymer_run("ACGTN")
    with pytest.raises(ValueError, match="cannot be empty"):
        longest_homopolymer_run("")


def test_base_entropy_is_maximal_for_even_composition():
    even = extract_quality_features(["ACGT" * 25]).base_entropy.iat[0]
    skewed = extract_quality_features(["A" * 100]).base_entropy.iat[0]
    assert even == pytest.approx(2.0)
    assert skewed == pytest.approx(0.0)


def test_entropy_does_not_warn_on_absent_bases():
    """np.where would still evaluate log2(0); the implementation must not."""
    with np.errstate(all="raise"):
        features = extract_quality_features(["A" * 50 + "C" * 50])
    assert np.isfinite(features.base_entropy.iat[0])
    assert np.isfinite(features.kmer_entropy.iat[0])


def test_homopolymer_artifact_moves_the_expected_features(clean_read):
    clean = extract_quality_features([clean_read]).iloc[0]
    artifact = extract_quality_features(["A" * 40 + clean_read[40:]]).iloc[0]

    assert artifact["longest_homopolymer"] > clean["longest_homopolymer"]
    assert artifact["kmer_entropy"] < clean["kmer_entropy"]
    assert artifact["max_kmer_frequency"] > clean["max_kmer_frequency"]
    assert artifact["distinct_kmers"] < clean["distinct_kmers"]


def test_adapter_contamination_is_visible_only_in_the_adapter_feature(clean_read):
    clean = extract_quality_features([clean_read]).iloc[0]
    contaminated = extract_quality_features(
        [clean_read[:40] + SEQUENCING_ADAPTER + clean_read[60:]]
    ).iloc[0]

    assert clean["adapter_8mer_hits"] == 0
    assert contaminated["adapter_8mer_hits"] > 0


def test_adapter_feature_follows_the_supplied_adapter():
    custom = "TTTTGGGGCCCCAAAA"
    hits = extract_quality_features([custom + CLEAN[16:]], adapter=custom)
    assert hits.adapter_8mer_hits.iat[0] > 0
    assert (
        extract_quality_features([CLEAN], adapter=custom).adapter_8mer_hits.iat[0]
        == 0
    )


def test_adapter_must_be_long_enough_to_match():
    with pytest.raises(ValueError, match="at least 8 bases"):
        extract_quality_features([CLEAN], adapter="ACGT")


def test_features_are_finite_across_the_real_dataset(dataset: pd.DataFrame):
    features = extract_quality_features(dataset["Sequence"].head(300))
    assert np.isfinite(features.to_numpy()).all()
    assert (features["gc_content"].between(0, 100)).all()
    assert (features["longest_homopolymer"] >= 1).all()


def test_sequences_are_normalized_before_use():
    spaced = extract_quality_features([" acgt " * 25]).iloc[0]
    plain = extract_quality_features([CLEAN]).iloc[0]
    pd.testing.assert_series_equal(spaced, plain, check_names=False)
