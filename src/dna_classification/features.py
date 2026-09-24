"""Feature engineering utilities for DNA sequence classification.

The module exposes two ways to build the model matrix:

- Functional helpers (:func:`extract_sequence_features`,
  :func:`fit_categorical_encoders`, :func:`build_feature_matrix`) for scripts
  and notebooks that manage the fit/transform split themselves.
- :class:`DnaFeatureExtractor`, a scikit-learn transformer that performs the
  same work inside a ``Pipeline``. Use it whenever the features feed
  cross-validation or a hyperparameter search, because it refits the
  categorical encoders on each training fold instead of on the full dataset.
"""

from itertools import product
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

DNA_BASES = ("A", "C", "G", "T")
KMER_SIZE = 3
DEFAULT_KMER_SIZES = (KMER_SIZE,)
POSITION_FEATURE_LENGTH = 100
CATEGORICAL_COLUMNS = ("Mutation_Flag", "Disease_Risk")


def kmer_labels(kmer_size: int) -> tuple[str, ...]:
    """Return every ordered DNA k-mer of length ``kmer_size``.

    Args:
        kmer_size: Length of each k-mer.

    Returns:
        A tuple of k-mer strings in lexicographic order.

    Raises:
        ValueError: If ``kmer_size`` is not a positive integer.
    """
    if not isinstance(kmer_size, int) or isinstance(kmer_size, bool):
        raise ValueError("Each k-mer size must be an integer.")
    if kmer_size < 1:
        raise ValueError(f"K-mer sizes must be positive; received {kmer_size}.")
    return tuple("".join(bases) for bases in product(DNA_BASES, repeat=kmer_size))


def normalize_kmer_sizes(kmer_sizes: Iterable[int] | int) -> tuple[int, ...]:
    """Validate and de-duplicate the requested k-mer sizes.

    Args:
        kmer_sizes: A single size or an iterable of sizes.

    Returns:
        A sorted tuple of distinct positive k-mer sizes.

    Raises:
        ValueError: If no size is supplied or any size is not a positive integer.
    """
    if isinstance(kmer_sizes, int) and not isinstance(kmer_sizes, bool):
        candidate_sizes: tuple[int, ...] = (kmer_sizes,)
    else:
        candidate_sizes = tuple(kmer_sizes)

    if not candidate_sizes:
        raise ValueError("At least one k-mer size is required.")
    for kmer_size in candidate_sizes:
        kmer_labels(kmer_size)
    return tuple(sorted(set(candidate_sizes)))


def encoded_column_name(column: str) -> str:
    """Return the model feature name produced by label-encoding ``column``."""
    return f"{column.lower()}_encoded"


def sequence_feature_names(
    kmer_sizes: Iterable[int] | int = DEFAULT_KMER_SIZES,
    include_positions: bool = True,
    position_length: int = POSITION_FEATURE_LENGTH,
) -> tuple[str, ...]:
    """Return the ordered sequence-feature schema for the given configuration.

    Args:
        kmer_sizes: K-mer lengths to expand into frequency features.
        include_positions: Whether to add per-position one-hot base indicators.
        position_length: Number of leading positions to encode.

    Returns:
        A tuple of feature names in the order the extractor produces them.
    """
    resolved_sizes = normalize_kmer_sizes(kmer_sizes)
    names = [
        "sequence_length",
        "gc_content",
        "at_content",
        *(f"{base}_count" for base in DNA_BASES),
        *(f"{base}_fraction" for base in DNA_BASES),
    ]
    for kmer_size in resolved_sizes:
        names.extend(f"kmer_{kmer}_frequency" for kmer in kmer_labels(kmer_size))
    if include_positions:
        names.extend(
            f"position_{position + 1}_{base}"
            for position in range(position_length)
            for base in DNA_BASES
        )
    return tuple(names)


SEQUENCE_FEATURE_NAMES = sequence_feature_names()
ENCODED_COLUMN_NAMES = tuple(
    encoded_column_name(column) for column in CATEGORICAL_COLUMNS
)
MODEL_FEATURE_NAMES = (*SEQUENCE_FEATURE_NAMES, *ENCODED_COLUMN_NAMES)
# Backwards-compatible alias for the sequence-only feature schema.
FEATURE_NAMES = SEQUENCE_FEATURE_NAMES
KMER_LABELS = kmer_labels(KMER_SIZE)


def normalize_sequence(sequence: str) -> str:
    """Normalize and validate a DNA sequence containing A, C, G, and T.

    Whitespace is removed and lowercase bases are converted to uppercase.

    Args:
        sequence: DNA sequence to validate.

    Returns:
        A whitespace-free, uppercase DNA sequence.

    Raises:
        TypeError: If ``sequence`` is not a string.
        ValueError: If the sequence is empty or contains unsupported symbols.
    """
    if not isinstance(sequence, str):
        raise TypeError("Each DNA sequence must be provided as a string.")

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("DNA sequences cannot be empty.")

    invalid_bases = sorted(set(normalized).difference(DNA_BASES))
    if invalid_bases:
        invalid_text = ", ".join(invalid_bases)
        raise ValueError(f"Unsupported DNA symbols: {invalid_text}.")

    return normalized


def extract_sequence_features(
    sequences: Iterable[str],
    kmer_sizes: Iterable[int] | int = DEFAULT_KMER_SIZES,
    include_positions: bool = True,
    position_length: int = POSITION_FEATURE_LENGTH,
) -> pd.DataFrame:
    """Convert DNA strings into composition and k-mer frequency features.

    Nucleotide bases are represented as composition frequencies, ordered k-mer
    frequencies, and one-hot indicators by position. This avoids assigning
    arbitrary numeric ranks to DNA symbols. Every feature is computed from a
    single sequence, so this step never mixes information across rows and is
    safe to apply before a train/test split.

    Args:
        sequences: Iterable of DNA strings.
        kmer_sizes: K-mer lengths to expand into frequency features.
        include_positions: Whether to add per-position one-hot base indicators.
        position_length: Number of leading positions to encode.

    Returns:
        A dataframe containing one numeric feature row per sequence.

    Raises:
        ValueError: If a k-mer size is invalid or any sequence is invalid.
    """
    resolved_sizes = normalize_kmer_sizes(kmer_sizes)
    column_names = sequence_feature_names(
        resolved_sizes, include_positions, position_length
    )
    all_kmer_labels = {
        kmer_size: kmer_labels(kmer_size) for kmer_size in resolved_sizes
    }

    feature_rows = []
    for sequence in sequences:
        normalized = normalize_sequence(sequence)
        sequence_length = len(normalized)
        base_counts = {base: normalized.count(base) for base in DNA_BASES}
        gc_count = base_counts["G"] + base_counts["C"]
        at_count = base_counts["A"] + base_counts["T"]

        row = {
            "sequence_length": sequence_length,
            "gc_content": 100.0 * gc_count / sequence_length,
            "at_content": 100.0 * at_count / sequence_length,
        }
        row.update({f"{base}_count": base_counts[base] for base in DNA_BASES})
        row.update(
            {
                f"{base}_fraction": base_counts[base] / sequence_length
                for base in DNA_BASES
            }
        )

        for kmer_size in resolved_sizes:
            kmer_counts = {kmer: 0 for kmer in all_kmer_labels[kmer_size]}
            for start in range(sequence_length - kmer_size + 1):
                kmer = normalized[start:start + kmer_size]
                if kmer in kmer_counts:
                    kmer_counts[kmer] += 1
            window_count = max(sequence_length - kmer_size + 1, 1)
            row.update(
                {
                    f"kmer_{kmer}_frequency": count / window_count
                    for kmer, count in kmer_counts.items()
                }
            )

        if include_positions:
            row.update(
                {
                    f"position_{position + 1}_{base}": int(
                        position < sequence_length
                        and normalized[position] == base
                    )
                    for position in range(position_length)
                    for base in DNA_BASES
                }
            )
        feature_rows.append(row)

    return pd.DataFrame(feature_rows, columns=list(column_names))


def fit_categorical_encoders(frame: pd.DataFrame) -> dict[str, dict]:
    """Fit one ``LabelEncoder`` per categorical column of the dataset.

    ``Mutation_Flag`` and ``Disease_Risk`` are categorical descriptors: the
    first is a binary flag and the second is an ordinal risk band. Both are
    stored as text categories and converted to integer codes, so no arbitrary
    arithmetic meaning is assumed before the model sees them. The modal
    category is recorded for each column and used whenever a value is missing
    or was never observed during fitting.

    This function learns from the rows it is given, so inside cross-validation
    it must see training-fold rows only. :class:`DnaFeatureExtractor` handles
    that automatically.

    Args:
        frame: Dataframe containing every column in ``CATEGORICAL_COLUMNS``.

    Returns:
        A mapping from column name to ``{"encoder", "default_code"}``.

    Raises:
        KeyError: If a categorical column is absent from ``frame``.
    """
    missing_columns = [
        column for column in CATEGORICAL_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing categorical columns: {missing_text}.")

    encoders: dict[str, dict] = {}
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype(str)
        encoder = LabelEncoder().fit(values)
        modal_value = values.mode().iat[0]
        encoders[column] = {
            "encoder": encoder,
            "default_code": int(encoder.transform([modal_value])[0]),
        }
    return encoders


def categorical_classes(encoders: Mapping[str, dict]) -> dict[str, list[str]]:
    """Return the categories each fitted encoder accepts, per column."""
    return {
        column: list(mapping["encoder"].classes_)
        for column, mapping in encoders.items()
    }


def encode_categorical_features(
    frame: pd.DataFrame,
    encoders: Mapping[str, dict],
) -> pd.DataFrame:
    """Label-encode the categorical columns using previously fitted encoders.

    Values that are missing, or that were not present when the encoders were
    fitted, fall back to the modal category recorded at fit time. This keeps
    inference inside the range of codes the model was trained on.

    Args:
        frame: Dataframe that may contain the categorical columns.
        encoders: Mapping produced by :func:`fit_categorical_encoders`.

    Returns:
        A dataframe of integer codes with one column per categorical feature.

    Raises:
        KeyError: If ``encoders`` is missing a fitted categorical column.
    """
    encoded_columns: dict[str, list[int]] = {}
    row_count = len(frame)
    for column in CATEGORICAL_COLUMNS:
        if column not in encoders:
            raise KeyError(f"No fitted encoder was supplied for {column}.")

        encoder = encoders[column]["encoder"]
        default_code = int(encoders[column]["default_code"])
        known_codes = {
            str(category): int(code)
            for code, category in enumerate(encoder.classes_)
        }
        if column in frame.columns:
            codes = [
                default_code
                if pd.isna(value)
                else known_codes.get(str(value), default_code)
                for value in frame[column]
            ]
        else:
            codes = [default_code] * row_count
        encoded_columns[encoded_column_name(column)] = codes

    return pd.DataFrame(encoded_columns, columns=list(ENCODED_COLUMN_NAMES))


def build_feature_matrix(
    frame: pd.DataFrame,
    encoders: Mapping[str, dict],
) -> pd.DataFrame:
    """Combine sequence-derived features with label-encoded categorical codes.

    Args:
        frame: Dataframe containing a ``Sequence`` column and, optionally, the
            categorical columns. Absent categorical columns are filled with
            their modal category.
        encoders: Mapping produced by :func:`fit_categorical_encoders`.

    Returns:
        A numeric dataframe whose columns follow ``MODEL_FEATURE_NAMES``.

    Raises:
        KeyError: If ``Sequence`` is missing from ``frame``.
    """
    if "Sequence" not in frame.columns:
        raise KeyError("The input dataframe must contain a Sequence column.")

    sequence_features = extract_sequence_features(frame["Sequence"])
    categorical_features = encode_categorical_features(frame, encoders)
    combined = pd.concat(
        [
            sequence_features.reset_index(drop=True),
            categorical_features.reset_index(drop=True),
        ],
        axis=1,
    )
    return combined[list(MODEL_FEATURE_NAMES)]


class DnaFeatureExtractor(BaseEstimator, TransformerMixin):
    """Turn a raw DNA dataframe into the numeric model matrix.

    The transformer accepts the raw dataframe (a ``Sequence`` column plus the
    optional categorical descriptors) and returns the full feature matrix. Only
    the categorical encoders are learned during ``fit``; every sequence feature
    is computed per row. Placing this step first in a ``Pipeline`` therefore
    guarantees that cross-validation and hyperparameter searches never fit an
    encoder on held-out rows.

    The feature-set options are constructor parameters, so a search can tune
    them alongside the model, for example ``features__kmer_sizes``.

    Args:
        kmer_sizes: K-mer lengths to expand into frequency features.
        include_positions: Whether to add per-position one-hot base indicators.
        position_length: Number of leading positions to encode.
        use_categorical: Whether to append the label-encoded categorical codes.
    """

    def __init__(
        self,
        kmer_sizes: Sequence[int] | int = DEFAULT_KMER_SIZES,
        include_positions: bool = True,
        position_length: int = POSITION_FEATURE_LENGTH,
        use_categorical: bool = True,
    ):
        self.kmer_sizes = kmer_sizes
        self.include_positions = include_positions
        self.position_length = position_length
        self.use_categorical = use_categorical

    def fit(self, X: pd.DataFrame, y=None) -> "DnaFeatureExtractor":
        """Fit the categorical encoders on the supplied rows only.

        Args:
            X: Raw dataframe with a ``Sequence`` column.
            y: Ignored, present for scikit-learn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            KeyError: If ``Sequence`` is missing, or the categorical columns are
                missing while ``use_categorical`` is enabled.
        """
        frame = self._validate_frame(X)
        self.encoders_ = (
            fit_categorical_encoders(frame) if self.use_categorical else {}
        )
        self.feature_names_ = list(
            sequence_feature_names(
                self.kmer_sizes, self.include_positions, self.position_length
            )
        )
        if self.use_categorical:
            self.feature_names_.extend(ENCODED_COLUMN_NAMES)
        self.n_features_in_ = len(self.feature_names_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the feature matrix for ``X`` using the fitted encoders.

        Args:
            X: Raw dataframe with a ``Sequence`` column.

        Returns:
            A numeric dataframe whose columns follow the fitted schema.

        Raises:
            AttributeError: If the transformer has not been fitted.
            KeyError: If ``Sequence`` is missing from ``X``.
        """
        if not hasattr(self, "feature_names_"):
            raise AttributeError(
                "DnaFeatureExtractor must be fitted before calling transform."
            )

        frame = self._validate_frame(X, require_categorical=False)
        features = extract_sequence_features(
            frame["Sequence"],
            self.kmer_sizes,
            self.include_positions,
            self.position_length,
        )
        if self.use_categorical:
            encoded = encode_categorical_features(frame, self.encoders_)
            features = pd.concat(
                [features.reset_index(drop=True), encoded.reset_index(drop=True)],
                axis=1,
            )
        return features[self.feature_names_]

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return the fitted feature names, for scikit-learn introspection."""
        if not hasattr(self, "feature_names_"):
            raise AttributeError(
                "DnaFeatureExtractor must be fitted before requesting names."
            )
        return list(self.feature_names_)

    def _validate_frame(
        self, X: pd.DataFrame, require_categorical: bool = True
    ) -> pd.DataFrame:
        """Check that ``X`` is a dataframe carrying the columns this step needs."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DnaFeatureExtractor expects a pandas DataFrame with a "
                "Sequence column."
            )
        if "Sequence" not in X.columns:
            raise KeyError("The input dataframe must contain a Sequence column.")
        if require_categorical and self.use_categorical:
            missing_columns = [
                column
                for column in CATEGORICAL_COLUMNS
                if column not in X.columns
            ]
            if missing_columns:
                missing_text = ", ".join(missing_columns)
                raise KeyError(f"Missing categorical columns: {missing_text}.")
        return X


# ---------------------------------------------------------------------------
# Quality-control features for unsupervised anomaly detection
# ---------------------------------------------------------------------------

# Prefix of the Illumina TruSeq adapter. Labs know which adapters they use, so
# encoding that knowledge as a feature is cheap and turns adapter
# contamination from nearly undetectable into nearly always detected.
SEQUENCING_ADAPTER = "AGATCGGAAGAGCACACGTC"
ADAPTER_MATCH_SIZE = 8

QUALITY_FEATURE_NAMES = (
    "gc_content",
    "base_entropy",
    "longest_homopolymer",
    "kmer_entropy",
    "max_kmer_frequency",
    "distinct_kmers",
    "adapter_8mer_hits",
)


def longest_homopolymer_run(sequence: str) -> int:
    """Return the length of the longest run of a single repeated base.

    Args:
        sequence: DNA sequence to scan.

    Returns:
        The length of the longest homopolymer run, at least 1.

    Raises:
        TypeError: If ``sequence`` is not a string.
        ValueError: If the sequence is empty or contains unsupported symbols.
    """
    normalized = normalize_sequence(sequence)
    best = current = 1
    for index in range(1, len(normalized)):
        if normalized[index] == normalized[index - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _shannon_entropy(frequencies) -> float:
    """Return the Shannon entropy in bits of a frequency vector.

    Zero-frequency entries are dropped before the logarithm rather than being
    masked afterwards: ``np.where`` evaluates both branches, so it would still
    compute ``log2(0)`` and emit divide-by-zero warnings.
    """
    import numpy as np

    positive = np.asarray(frequencies, dtype=float)
    positive = positive[positive > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def extract_quality_features(
    sequences: Iterable[str],
    adapter: str = SEQUENCING_ADAPTER,
) -> pd.DataFrame:
    """Build compact per-read features for sequencing quality control.

    The seven features are chosen to expose the artifacts a sequencing run
    produces: composition shifts (``gc_content``), low complexity
    (``base_entropy``, ``kmer_entropy``, ``distinct_kmers``), repeat expansions
    (``longest_homopolymer``, ``max_kmer_frequency``), and residual adapter
    (``adapter_8mer_hits``).

    Unlike :func:`extract_sequence_features`, this returns a small,
    interpretable matrix suited to distance- and density-based outlier
    detection, which degrades badly in hundreds of dimensions.

    Args:
        sequences: Iterable of DNA strings.
        adapter: Known adapter sequence to count overlapping matches against.

    Returns:
        A dataframe with one row per read and ``QUALITY_FEATURE_NAMES`` columns.

    Raises:
        ValueError: If a sequence is invalid or the adapter is too short.
    """
    import numpy as np

    if len(adapter) < ADAPTER_MATCH_SIZE:
        raise ValueError(
            f"The adapter must be at least {ADAPTER_MATCH_SIZE} bases long."
        )
    adapter_kmers = {
        adapter[i:i + ADAPTER_MATCH_SIZE]
        for i in range(len(adapter) - ADAPTER_MATCH_SIZE + 1)
    }

    rows = []
    for sequence in sequences:
        normalized = normalize_sequence(sequence)
        length = len(normalized)
        base_counts = {base: normalized.count(base) for base in DNA_BASES}
        base_fractions = [base_counts[base] / length for base in DNA_BASES]

        kmer_counts = {kmer: 0 for kmer in KMER_LABELS}
        for start in range(length - KMER_SIZE + 1):
            kmer = normalized[start:start + KMER_SIZE]
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
        total_kmers = max(sum(kmer_counts.values()), 1)
        kmer_frequencies = np.array(
            [count / total_kmers for count in kmer_counts.values()]
        )

        adapter_hits = sum(
            1
            for i in range(length - ADAPTER_MATCH_SIZE + 1)
            if normalized[i:i + ADAPTER_MATCH_SIZE] in adapter_kmers
        )

        rows.append(
            {
                "gc_content": 100.0
                * (base_counts["G"] + base_counts["C"])
                / length,
                "base_entropy": _shannon_entropy(base_fractions),
                "longest_homopolymer": longest_homopolymer_run(normalized),
                "kmer_entropy": _shannon_entropy(kmer_frequencies),
                "max_kmer_frequency": float(kmer_frequencies.max()),
                "distinct_kmers": int((kmer_frequencies > 0).sum()),
                "adapter_8mer_hits": adapter_hits,
            }
        )

    return pd.DataFrame(rows, columns=list(QUALITY_FEATURE_NAMES))
