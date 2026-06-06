from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_utils import dataset_summary, load_news_dataset


LABELS = [0, 1]
TARGET_NAMES = ["True", "Fake"]


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
    )


def make_baseline_models(max_features: int, random_state: int) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", build_vectorizer(max_features)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", build_vectorizer(max_features)),
                ("model", MultinomialNB()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("tfidf", build_vectorizer(max_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def make_stacking_model(max_features: int, random_state: int, cv_splits: int) -> Pipeline:
    estimators = [
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ("nb", MultinomialNB()),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=120,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        ),
    ]
    return Pipeline(
        [
            ("tfidf", build_vectorizer(max_features)),
            (
                "model",
                StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
                    cv=cv_splits,
                    n_jobs=-1,
                    stack_method="auto",
                ),
            ),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def save_confusion_matrix(y_test: pd.Series, predictions, reports_dir: Path) -> None:
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    pd.DataFrame(matrix, index=TARGET_NAMES, columns=TARGET_NAMES).to_csv(reports_dir / "confusion_matrix.csv")
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=TARGET_NAMES)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Stacking Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def write_summary_markdown(
    reports_dir: Path,
    data_summary: dict[str, object],
    comparison: pd.DataFrame,
    best_metrics: dict[str, float | str],
) -> None:
    lines = [
        "# Task III Model Evaluation Summary",
        "",
        "## Dataset",
        f"- Rows used: {data_summary['rows']}",
        f"- Class distribution: {data_summary['class_distribution']}",
        f"- Average cleaned text length: {data_summary['average_text_length']:.1f} characters",
        "",
        "## Model Comparison",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Selected Model",
        "- Final application model: Stacking ensemble over Logistic Regression, Multinomial Naive Bayes, and Random Forest.",
        f"- Test F1: {float(best_metrics['f1']):.4f}",
        "",
        "## Interpretation",
        "- Logistic Regression handles sparse TF-IDF features and gives a strong linear baseline.",
        "- Multinomial Naive Bayes provides a fast probabilistic text baseline.",
        "- Random Forest adds non-linear interactions.",
        "- Stacking learns how to combine the base models instead of relying on a fixed majority vote.",
    ]
    (reports_dir / "training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Task III fake news detection ensemble.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--sample-path", default="data/sample_news.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-sample-if-missing", action="store_true")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    frame = load_news_dataset(args.data_dir, args.sample_path, use_sample_if_missing=args.use_sample_if_missing)
    if frame["label"].nunique() != 2:
        raise ValueError("Training requires both True and Fake classes.")

    class_counts = frame["label"].value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 3:
        raise ValueError("Each class needs at least 3 records for stratified training and validation.")

    x_train, x_test, y_train, y_test = train_test_split(
        frame["combined_text"],
        frame["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=frame["label"],
    )

    train_min_class_count = int(y_train.value_counts().min())
    cv_splits = max(2, min(5, train_min_class_count))
    baselines = make_baseline_models(args.max_features, args.random_state)

    rows = []
    for name, model in baselines.items():
        model.fit(x_train, y_train)
        rows.append(evaluate_model(name, model, x_test, y_test))

    final_model = make_stacking_model(args.max_features, args.random_state, cv_splits)
    final_model.fit(x_train, y_train)
    final_predictions = final_model.predict(x_test)
    final_metrics = evaluate_model("Stacking Ensemble", final_model, x_test, y_test)
    rows.append(final_metrics)

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    report_text = classification_report(y_test, final_predictions, labels=LABELS, target_names=TARGET_NAMES, zero_division=0)
    (reports_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrix(y_test, final_predictions, reports_dir)

    metrics_summary = {
        "dataset": dataset_summary(frame),
        "cv_splits": cv_splits,
        "test_size": args.test_size,
        "selected_model": "Stacking Ensemble",
        "selected_metrics": {key: value for key, value in final_metrics.items() if key != "model"},
    }
    (reports_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    write_summary_markdown(reports_dir, metrics_summary["dataset"], comparison, final_metrics)

    joblib.dump(final_model, models_dir / "fake_news_stacking_pipeline.joblib")
    (models_dir / "label_mapping.json").write_text(json.dumps({"0": "True", "1": "Fake"}, indent=2), encoding="utf-8")

    print("Training complete")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
