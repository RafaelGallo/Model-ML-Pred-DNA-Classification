"""Tests for sequence feature extraction and categorical label encoding."""

import pandas as pd
import pytest

from sklearn.base import clone

from dna_classification.features import (
    CATEGORICAL_COLUMNS,
    ENCODED_COLUMN_NAMES,
    MODEL_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    DnaFeatureExtractor,
    build_feature_matrix,
    categorical_classes,
    encode_categorical_features,
    extract_sequence_features,
    fit_categorical_encoders,
    kmer_labels,
    normalize_kmer_sizes,
    normalize_sequence,
    sequence_feature_names,
)

SEQUENCE_A = "ACGT" * 25
SEQUENCE_B = "AACCGGTT" * 12 + "ACGT"


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    """Return a small frame with both sequences and categorical descriptors."""
    return pd.DataFrame(
        {
            "Sequence": [SEQUENCE_A, SEQUENCE_B],
            "Mutation_Flag": [0, 1],
            "Disease_Risk": ["Low", "High"],
        }
    )


@pytest.fixture()
def encoders(sample_frame: pd.DataFrame) -> dict:
    """Fit encoders on a frame containing every category of interest."""
    fitting_frame = pd.DataFrame(
        {
            "Mutation_Flag": [0, 0, 0, 1],
            "Disease_Risk": ["Low", "Low", "Medium", "High"],
        }
    )
    return fit_categorical_encoders(fitting_frame)


def test_normalize_sequence_uppercases_and_strips_whitespace():
    assert normalize_sequence(" acg t\n") == "ACGT"


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_normalize_sequence_rejects_empty_input(value):
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_sequence(value)


def test_normalize_sequence_rejects_unsupported_symbols():
    with pytest.raises(ValueError, match="Unsupported DNA symbols: N, X"):
        normalize_sequence("ACGTNX")


def test_normalize_sequence_rejects_non_string():
    with pytest.raises(TypeError, match="must be provided as a string"):
        normalize_sequence(42)


def test_extract_sequence_features_uses_the_declared_schema():
    features = extract_sequence_features([SEQUENCE_A])
    assert list(features.columns) == list(SEQUENCE_FEATURE_NAMES)
    assert len(features) == 1


def test_extract_sequence_features_computes_composition():
    features = extract_sequence_features(["AACCGG"]).iloc[0]
    assert features["sequence_length"] == 6
    assert features["gc_content"] == pytest.approx(200 / 3)
    assert features["at_content"] == pytest.approx(100 / 3)
    assert features["A_count"] == 2
    assert features["T_count"] == 0
    assert features["C_fraction"] == pytest.approx(1 / 3)


def test_kmer_frequencies_sum_to_one():
    features = extract_sequence_features([SEQUENCE_A, SEQUENCE_B])
    kmer_columns = [
        name for name in SEQUENCE_FEATURE_NAMES if name.startswith("kmer_")
    ]
    totals = features[kmer_columns].sum(axis=1)
    assert totals.to_numpy() == pytest.approx([1.0, 1.0])


def test_position_indicators_are_one_hot():
    features = extract_sequence_features([SEQUENCE_A]).iloc[0]
    assert features["position_1_A"] == 1
    assert features["position_1_C"] == 0
    assert features["position_2_C"] == 1
    position_columns = [
        name for name in SEQUENCE_FEATURE_NAMES if name.startswith("position_")
    ]
    assert features[position_columns].sum() == 100


@pytest.mark.parametrize("invalid_size", [0, -1, 1.5])
def test_extract_sequence_features_rejects_invalid_kmer_sizes(invalid_size):
    with pytest.raises(ValueError):
        extract_sequence_features([SEQUENCE_A], kmer_sizes=(invalid_size,))


def test_extract_sequence_features_requires_at_least_one_kmer_size():
    with pytest.raises(ValueError, match="At least one k-mer size"):
        extract_sequence_features([SEQUENCE_A], kmer_sizes=())


def test_kmer_labels_cover_every_combination():
    assert len(kmer_labels(1)) == 4
    assert len(kmer_labels(3)) == 64
    assert kmer_labels(2)[0] == "AA"


def test_normalize_kmer_sizes_sorts_and_deduplicates():
    assert normalize_kmer_sizes([3, 2, 3]) == (2, 3)
    assert normalize_kmer_sizes(4) == (4,)


def test_multiple_kmer_sizes_extend_the_schema():
    names = sequence_feature_names(kmer_sizes=(2, 3), include_positions=False)
    assert sum(name.startswith("kmer_") for name in names) == 16 + 64
    features = extract_sequence_features(
        [SEQUENCE_A], kmer_sizes=(2, 3), include_positions=False
    )
    assert list(features.columns) == list(names)


def test_positions_can_be_disabled():
    features = extract_sequence_features([SEQUENCE_A], include_positions=False)
    assert not any(name.startswith("position_") for name in features.columns)


def test_default_schema_is_unchanged_by_the_configurable_arguments():
    """The saved model bundle depends on this exact default column order."""
    features = extract_sequence_features([SEQUENCE_A])
    assert list(features.columns) == list(SEQUENCE_FEATURE_NAMES)
    assert len(SEQUENCE_FEATURE_NAMES) == 475


def test_fit_categorical_encoders_requires_every_column():
    with pytest.raises(KeyError, match="Disease_Risk"):
        fit_categorical_encoders(pd.DataFrame({"Mutation_Flag": [0, 1]}))


def test_categorical_classes_are_sorted_alphabetically(encoders):
    assert categorical_classes(encoders) == {
        "Mutation_Flag": ["0", "1"],
        "Disease_Risk": ["High", "Low", "Medium"],
    }


def test_encode_categorical_features_maps_known_categories(encoders, sample_frame):
    encoded = encode_categorical_features(sample_frame, encoders)
    assert list(encoded.columns) == list(ENCODED_COLUMN_NAMES)
    assert encoded["mutation_flag_encoded"].tolist() == [0, 1]
    assert encoded["disease_risk_encoded"].tolist() == [1, 0]


def test_unseen_and_missing_values_fall_back_to_the_modal_category(encoders):
    frame = pd.DataFrame(
        {
            "Sequence": [SEQUENCE_A, SEQUENCE_A],
            "Mutation_Flag": [None, 7],
            "Disease_Risk": ["Unknown", None],
        }
    )
    encoded = encode_categorical_features(frame, encoders)
    # Modal categories of the fitting frame are Mutation_Flag "0" and
    # Disease_Risk "Low", encoded as 0 and 1.
    assert encoded["mutation_flag_encoded"].tolist() == [0, 0]
    assert encoded["disease_risk_encoded"].tolist() == [1, 1]


def test_absent_categorical_columns_fall_back_to_the_modal_category(encoders):
    encoded = encode_categorical_features(
        pd.DataFrame({"Sequence": [SEQUENCE_A]}), encoders
    )
    assert encoded["mutation_flag_encoded"].tolist() == [0]
    assert encoded["disease_risk_encoded"].tolist() == [1]


def test_encode_categorical_features_requires_a_fitted_encoder(sample_frame):
    with pytest.raises(KeyError, match="Mutation_Flag"):
        encode_categorical_features(sample_frame, {})


def test_build_feature_matrix_follows_the_model_schema(encoders, sample_frame):
    features = build_feature_matrix(sample_frame, encoders)
    assert list(features.columns) == list(MODEL_FEATURE_NAMES)
    assert len(features) == len(sample_frame)
    assert features.notna().all().all()


def test_build_feature_matrix_works_without_categorical_columns(encoders):
    features = build_feature_matrix(
        pd.DataFrame({"Sequence": [SEQUENCE_A]}), encoders
    )
    assert list(features.columns) == list(MODEL_FEATURE_NAMES)


def test_build_feature_matrix_requires_a_sequence_column(encoders):
    with pytest.raises(KeyError, match="Sequence column"):
        build_feature_matrix(pd.DataFrame({"Mutation_Flag": [0]}), encoders)


def test_model_schema_extends_the_sequence_schema():
    assert MODEL_FEATURE_NAMES == (*SEQUENCE_FEATURE_NAMES, *ENCODED_COLUMN_NAMES)
    assert len(ENCODED_COLUMN_NAMES) == len(CATEGORICAL_COLUMNS)


def test_extractor_transform_matches_the_functional_path(encoders, sample_frame):
    extractor = DnaFeatureExtractor().fit(sample_frame)
    from_transformer = extractor.transform(sample_frame)
    from_function = build_feature_matrix(sample_frame, extractor.encoders_)
    pd.testing.assert_frame_equal(from_transformer, from_function)


def test_extractor_fits_encoders_only_on_the_rows_it_is_given():
    """This is what keeps cross-validation folds free of encoder leakage."""
    training_fold = pd.DataFrame(
        {
            "Sequence": [SEQUENCE_A, SEQUENCE_B],
            "Mutation_Flag": [0, 0],
            "Disease_Risk": ["Low", "Low"],
        }
    )
    extractor = DnaFeatureExtractor().fit(training_fold)
    assert list(extractor.encoders_["Disease_Risk"]["encoder"].classes_) == ["Low"]

    held_out = pd.DataFrame(
        {
            "Sequence": [SEQUENCE_A],
            "Mutation_Flag": [1],
            "Disease_Risk": ["High"],
        }
    )
    encoded = extractor.transform(held_out)
    # Unseen categories collapse onto the modal training code rather than
    # silently inventing a new one.
    assert encoded["disease_risk_encoded"].tolist() == [0]
    assert encoded["mutation_flag_encoded"].tolist() == [0]


def test_extractor_reports_its_feature_names(sample_frame):
    extractor = DnaFeatureExtractor().fit(sample_frame)
    assert extractor.get_feature_names_out() == list(MODEL_FEATURE_NAMES)
    assert extractor.n_features_in_ == len(MODEL_FEATURE_NAMES)


def test_extractor_can_drop_the_categorical_block(sample_frame):
    extractor = DnaFeatureExtractor(use_categorical=False).fit(sample_frame)
    features = extractor.transform(sample_frame)
    assert list(features.columns) == list(SEQUENCE_FEATURE_NAMES)
    assert extractor.encoders_ == {}


def test_extractor_honours_the_feature_configuration(sample_frame):
    extractor = DnaFeatureExtractor(
        kmer_sizes=(2, 3), include_positions=False, use_categorical=False
    ).fit(sample_frame)
    features = extractor.transform(sample_frame)
    assert features.shape[1] == 11 + 16 + 64


def test_extractor_is_clonable_with_its_parameters():
    extractor = DnaFeatureExtractor(kmer_sizes=(2, 3), include_positions=False)
    copy = clone(extractor)
    assert copy.get_params() == extractor.get_params()


def test_extractor_requires_fitting_before_transform(sample_frame):
    with pytest.raises(AttributeError, match="must be fitted"):
        DnaFeatureExtractor().transform(sample_frame)


def test_extractor_rejects_input_without_a_sequence_column():
    with pytest.raises(KeyError, match="Sequence column"):
        DnaFeatureExtractor().fit(pd.DataFrame({"Mutation_Flag": [0]}))


def test_extractor_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        DnaFeatureExtractor().fit([SEQUENCE_A])


def test_extractor_requires_categorical_columns_when_enabled():
    with pytest.raises(KeyError, match="Missing categorical columns"):
        DnaFeatureExtractor().fit(pd.DataFrame({"Sequence": [SEQUENCE_A]}))


def test_extractor_transform_tolerates_missing_categorical_columns(sample_frame):
    extractor = DnaFeatureExtractor().fit(sample_frame)
    features = extractor.transform(pd.DataFrame({"Sequence": [SEQUENCE_A]}))
    assert list(features.columns) == list(MODEL_FEATURE_NAMES)
