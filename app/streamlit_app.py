"""Streamlit interface for unsupervised sequencing quality control.

The app screens DNA reads with the Isolation Forest detector trained in
`notebooks/05_dna_unsupervised_learning.ipynb`. It answers one operational
question: given a manual review budget, which reads should a technician look at?
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "src"
if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))

from dna_classification.features import extract_quality_features  # noqa: E402
from dna_classification.features import normalize_sequence  # noqa: E402

DETECTOR_PATH = PROJECT_ROOT / "models" / "dna_anomaly_detector.joblib"
ARTIFACT_LABELS = {
    "adapter": "Adapter contamination",
    "homopolymer": "Homopolymer artifact",
    "gc_shift": "GC-shifted contaminant",
    "chimera": "Chimeric read",
}


@st.cache_resource
def load_detector(detector_path: str) -> dict:
    """Load the saved detector bundle once per Streamlit session."""
    return joblib.load(detector_path)


def score_reads(bundle: dict, sequences: list[str]) -> pd.DataFrame:
    """Return the anomaly score and quality features for each read.

    Sequences are validated first so an unsupported symbol produces a clear
    message rather than a failure deep inside the feature builder.
    """
    normalized = [normalize_sequence(sequence) for sequence in sequences]
    features = extract_quality_features(normalized, adapter=bundle["adapter"])
    scores = -bundle["pipeline"].score_samples(features)

    results = features.copy()
    results.insert(0, "anomaly_score", scores)
    return results


def budget_threshold(bundle: dict, budget: float) -> float:
    """Return the score threshold calibrated for a review budget."""
    thresholds = bundle["score_thresholds"]
    if budget in thresholds:
        return thresholds[budget]
    closest = min(thresholds, key=lambda candidate: abs(candidate - budget))
    return thresholds[closest]


def render_sidebar(bundle: dict) -> float:
    """Show what the detector is and let the reviewer pick a budget."""
    metrics = bundle.get("holdout_metrics", {})
    with st.sidebar:
        st.subheader("Detector")
        st.write(bundle["model_name"])
        st.caption(
            f"Unsupervised: fitted on {bundle['training_reads']:,} clean reads "
            f"with no anomaly labels. Selected as the "
            f"{bundle.get('selection', 'best detector')}."
        )
        if metrics:
            roc_auc = metrics.get("roc_auc")
            if roc_auc is not None:
                st.metric("Held-out ROC AUC", f"{roc_auc:.3f}")
            recall_5 = metrics.get("recall_at_5pct")
            if recall_5 is not None:
                st.metric("Artifacts caught at a 5% budget", f"{recall_5:.0%}")

        st.subheader("Review budget")
        budget = st.select_slider(
            "Share of reads sent to manual review",
            options=sorted(bundle["score_thresholds"]),
            value=0.05,
            format_func=lambda value: f"{value:.0%}",
        )
        st.caption(
            f"Reads scoring above {budget_threshold(bundle, budget):.4f} are "
            "flagged. The threshold is calibrated on clean reads, so a batch "
            "carrying contamination flags slightly more than the budget."
        )

        recall_by_artifact = bundle.get("recall_by_artifact_at_5pct", {})
        if recall_by_artifact:
            st.subheader("Detection by artifact")
            st.caption("Recall at a 5% review budget, measured on held-out data.")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Artifact": [
                            ARTIFACT_LABELS.get(kind, kind)
                            for kind in recall_by_artifact
                        ],
                        "Caught": list(recall_by_artifact.values()),
                    }
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Caught": st.column_config.ProgressColumn(
                        "Caught", format="%.0f%%", min_value=0.0, max_value=1.0
                    )
                },
            )
        undetectable = bundle.get("undetectable_artifacts", [])
        if undetectable:
            names = ", ".join(
                f"{ARTIFACT_LABELS.get(kind, kind).lower()}s"
                for kind in undetectable
            )
            st.warning(
                f"This detector cannot find {names}. Composition features look "
                "normal for those reads; catching them needs alignment to a "
                "reference."
            )
    return budget


def render_single_read_tab(bundle: dict, budget: float) -> None:
    """Screen one pasted read and explain the verdict."""
    sequence = st.text_area(
        "DNA read",
        placeholder="Paste a read containing A, C, G, and T.",
        height=140,
    )
    if not st.button("Screen read", type="primary"):
        return

    try:
        result = score_reads(bundle, [sequence]).iloc[0]
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return

    threshold = budget_threshold(bundle, budget)
    flagged = result["anomaly_score"] > threshold

    if flagged:
        st.error("Flag for manual review")
    else:
        st.success("Pass")
    st.metric(
        "Anomaly score",
        f"{result['anomaly_score']:.4f}",
        delta=f"{result['anomaly_score'] - threshold:+.4f} vs threshold",
        delta_color="inverse",
    )

    st.subheader("Quality features")
    st.caption(
        "These seven features are the whole input to the detector. A value far "
        "from the clean-read norm is what drives a high score."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": bundle["feature_names"],
                "Value": [float(result[name]) for name in bundle["feature_names"]],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_batch_tab(bundle: dict, budget: float) -> None:
    """Screen an uploaded batch and return the reads worth reviewing."""
    st.caption("The file must contain a Sequence column.")
    uploaded_file = st.file_uploader("Upload a CSV of reads", type=["csv"])
    if uploaded_file is None:
        return

    try:
        input_frame = pd.read_csv(uploaded_file)
        if "Sequence" not in input_frame.columns:
            raise ValueError("The uploaded CSV must contain a Sequence column.")

        results = score_reads(bundle, input_frame["Sequence"].astype(str).tolist())
        threshold = budget_threshold(bundle, budget)
        results.insert(0, "flagged", results["anomaly_score"] > threshold)
        identifier = (
            input_frame["Sample_ID"]
            if "Sample_ID" in input_frame.columns
            else input_frame["Sequence"].str[:20].add("...")
        )
        results.insert(0, "read", identifier.to_numpy())
        results = results.sort_values("anomaly_score", ascending=False)

        flagged_count = int(results["flagged"].sum())
        first, second = st.columns(2)
        first.metric("Reads screened", f"{len(results):,}")
        second.metric(
            "Flagged for review",
            f"{flagged_count:,}",
            delta=f"{flagged_count / len(results):.1%} of the batch",
        )
        if flagged_count / len(results) > budget * 1.5:
            st.warning(
                "This batch is flagging well above the budget, which usually "
                "means genuinely elevated contamination rather than a "
                "threshold problem."
            )

        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download flagged reads",
            data=results[results["flagged"]].to_csv(index=False).encode("utf-8"),
            file_name="qc_flagged_reads.csv",
            mime="text/csv",
        )
    except (KeyError, TypeError, ValueError, pd.errors.ParserError) as error:
        st.error(str(error))


def main() -> None:
    """Render the quality-control screening interface."""
    st.set_page_config(page_title="DNA Quality Control", page_icon=None)
    st.title("DNA Sequencing Quality Control")
    st.write(
        "Rank incoming DNA reads by how unusual they are, so limited manual "
        "review time goes to the reads most likely to be contaminated or "
        "artifactual. The detector is unsupervised: it learned what a clean "
        "read looks like and never saw a labeled anomaly."
    )

    if not DETECTOR_PATH.exists():
        st.error(
            "The detector was not found. Run "
            "notebooks/05_dna_unsupervised_learning.ipynb to create "
            "models/dna_anomaly_detector.joblib."
        )
        st.stop()

    bundle = load_detector(str(DETECTOR_PATH))
    budget = render_sidebar(bundle)

    single_tab, batch_tab = st.tabs(["Single read", "Batch CSV"])
    with single_tab:
        render_single_read_tab(bundle, budget)
    with batch_tab:
        render_batch_tab(bundle, budget)


if __name__ == "__main__":
    main()
