from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "fake_news_stacking_pipeline.joblib"
METRICS_PATH = APP_DIR / "reports" / "metrics_summary.json"
COMPARISON_PATH = APP_DIR / "reports" / "model_comparison.csv"
CONFUSION_MATRIX_PATH = APP_DIR / "reports" / "confusion_matrix.png"


SAMPLES = {
    "Likely true public-health report": (
        "The city health department reported that vaccination clinics will operate at community centers this weekend. "
        "Officials said appointment numbers, staffing levels, and vaccine inventory will be published daily on the city website."
    ),
    "Likely fabricated viral claim": (
        "A secret document proves that every major news agency is controlled by one hidden committee, according to an anonymous "
        "post that offered no source, date, named witness, or verifiable document link."
    ),
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def top_input_terms(model, text: str, limit: int = 8) -> list[str]:
    try:
        vectorizer = model.named_steps["tfidf"]
        matrix = vectorizer.transform([text])
        values = matrix.toarray()[0]
        if not np.any(values):
            return []
        feature_names = np.array(vectorizer.get_feature_names_out())
        top_indices = values.argsort()[::-1][:limit]
        return [feature_names[index] for index in top_indices if values[index] > 0]
    except Exception:
        return []


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_page(model) -> None:
    st.header("Narrative Classification")
    sample_choice = st.selectbox("Load an example", [""] + list(SAMPLES.keys()))
    default_text = SAMPLES.get(sample_choice, "")
    article_text = st.text_area("News article or claim", value=default_text, height=220)

    if st.button("Classify narrative", type="primary"):
        if model is None:
            st.error("Model file is missing. Run `python train_model.py --use-sample-if-missing` first.")
            return
        if len(article_text.strip()) < 20:
            st.warning("Enter at least one complete sentence.")
            return

        prediction = int(model.predict([article_text])[0])
        probabilities = model.predict_proba([article_text])[0]
        fake_probability = float(probabilities[1])
        true_probability = float(probabilities[0])
        label = "Fabricated / Fake" if prediction == 1 else "Likely True"

        st.metric("Prediction", label)
        col_a, col_b = st.columns(2)
        col_a.metric("Fake probability", f"{fake_probability:.1%}")
        col_b.metric("True probability", f"{true_probability:.1%}")

        terms = top_input_terms(model, article_text)
        if terms:
            st.caption("High-signal TF-IDF terms in this input")
            st.write(", ".join(terms))


def performance_page() -> None:
    st.header("Model Performance")
    metrics = load_json(METRICS_PATH)
    if metrics:
        selected = metrics.get("selected_metrics", {})
        cols = st.columns(4)
        for column, key in zip(cols, ["accuracy", "precision", "recall", "f1"]):
            column.metric(key.upper(), f"{float(selected.get(key, 0)):.3f}")
        st.json(metrics.get("dataset", {}), expanded=False)

    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
        st.subheader("Baseline Comparison")
        st.dataframe(comparison, use_container_width=True)

    if CONFUSION_MATRIX_PATH.exists():
        st.subheader("Confusion Matrix")
        st.image(str(CONFUSION_MATRIX_PATH))


def testing_page() -> None:
    st.header("Stakeholder Testing Protocol")
    st.markdown(
        """
        1. Ask a target user to classify 5-10 short news items manually.
        2. Run the same items through this prototype.
        3. Record whether the prediction was useful, confusing, or missing context.
        4. Use the feedback to improve the interface explanation and retraining plan.
        """
    )
    st.info("The included sample log is a demonstration template only. Replace it with real participant feedback before final submission.")


def main() -> None:
    st.set_page_config(page_title="Fabricated Narrative Detector", layout="wide")
    st.title("Multi-Stage Ensemble Framework for Fabricated Digital Narratives")
    st.caption("Task III prototype: TF-IDF preprocessing, baseline models, stacking ensemble, and stakeholder testing workflow.")

    model = load_model()
    page = st.sidebar.radio("View", ["Classify", "Performance", "Stakeholder Testing"])
    if page == "Classify":
        prediction_page(model)
    elif page == "Performance":
        performance_page()
    else:
        testing_page()


if __name__ == "__main__":
    main()
