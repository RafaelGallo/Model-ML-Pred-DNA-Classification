"""Tests for the structure and quality of the source dataset."""

import pandas as pd

EXPECTED_COLUMNS = [
    "Sample_ID",
    "Sequence",
    "GC_Content",
    "AT_Content",
    "Sequence_Length",
    "Num_A",
    "Num_T",
    "Num_C",
    "Num_G",
    "kmer_3_freq",
    "Mutation_Flag",
    "Class_Label",
    "Disease_Risk",
]
EXPECTED_CLASSES = {"Bacteria", "Human", "Plant", "Virus"}


def test_dataset_has_the_documented_columns(dataset: pd.DataFrame):
    assert list(dataset.columns) == EXPECTED_COLUMNS


def test_dataset_has_three_thousand_unique_samples(dataset: pd.DataFrame):
    assert len(dataset) == 3_000
    assert dataset["Sample_ID"].nunique() == len(dataset)


def test_dataset_has_no_missing_values_or_duplicate_rows(dataset: pd.DataFrame):
    assert dataset.isna().sum().sum() == 0
    assert dataset.duplicated().sum() == 0


def test_target_contains_only_the_four_documented_classes(dataset: pd.DataFrame):
    assert set(dataset["Class_Label"]) == EXPECTED_CLASSES


def test_sequences_use_only_canonical_bases(dataset: pd.DataFrame):
    observed_alphabet = set("".join(dataset["Sequence"].astype(str)))
    assert observed_alphabet == {"A", "C", "G", "T"}


def test_reported_sequence_length_matches_the_actual_sequence(dataset: pd.DataFrame):
    assert (dataset["Sequence"].str.len() == dataset["Sequence_Length"]).all()


def test_supplied_summaries_are_internally_consistent(dataset: pd.DataFrame):
    base_totals = dataset[["Num_A", "Num_T", "Num_C", "Num_G"]].sum(axis=1)
    assert (base_totals == dataset["Sequence_Length"]).all()
    content_totals = dataset["GC_Content"] + dataset["AT_Content"]
    assert content_totals.round(6).eq(100.0).all()


def test_categorical_descriptors_use_the_documented_values(dataset: pd.DataFrame):
    assert set(dataset["Mutation_Flag"]) == {0, 1}
    assert set(dataset["Disease_Risk"]) == {"Low", "Medium", "High"}


def test_classes_are_close_to_balanced(dataset: pd.DataFrame):
    shares = dataset["Class_Label"].value_counts(normalize=True)
    assert shares.between(0.22, 0.28).all()
