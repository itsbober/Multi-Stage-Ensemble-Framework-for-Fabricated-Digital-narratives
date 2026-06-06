from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


TEXT_COLUMNS = ("text", "content", "article", "body", "news", "statement")
TITLE_COLUMNS = ("title", "headline", "subject")
LABEL_COLUMNS = ("label", "class", "target", "category", "type", "is_fake")


def clean_text(value: object) -> str:
    """Normalize article text without removing the linguistic cues used by TF-IDF."""
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s.,;:!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _map_label(value: object, label_column: str | None = None) -> int:
    if value is None or pd.isna(value):
        raise ValueError("Missing label value")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = int(value)
        if numeric in (0, 1):
            if label_column and label_column.lower() == "is_fake":
                return numeric
            return numeric

    normalized = str(value).strip().lower()
    if normalized in {"fake", "fabricated", "false", "misinformation", "disinformation", "unreliable", "1", "yes", "y", "f"}:
        return 1
    if normalized in {"true", "real", "reliable", "genuine", "0", "no", "n", "t"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def _standardize_frame(frame: pd.DataFrame, label: int | None = None, source_name: str = "") -> pd.DataFrame:
    text_col = _first_existing(frame.columns, TEXT_COLUMNS)
    title_col = _first_existing(frame.columns, TITLE_COLUMNS)
    label_col = _first_existing(frame.columns, LABEL_COLUMNS)

    if text_col is None and title_col is None:
        raise ValueError(f"{source_name or 'CSV'} must contain at least one title/text column")

    output = pd.DataFrame()
    output["title"] = frame[title_col].fillna("").astype(str) if title_col else ""
    output["text"] = frame[text_col].fillna("").astype(str) if text_col else ""
    output["combined_text"] = (output["title"] + " " + output["text"]).map(clean_text)

    if label is not None:
        output["label"] = label
    elif label_col:
        output["label"] = frame[label_col].map(lambda value: _map_label(value, label_col))
    else:
        raise ValueError(f"{source_name or 'CSV'} must contain a label column")

    output = output[output["combined_text"].str.len() >= 20].copy()
    output = output.drop_duplicates(subset=["combined_text"])
    output["label_name"] = output["label"].map({0: "True", 1: "Fake"})
    return output.reset_index(drop=True)


def _load_true_fake_pair(data_dir: Path) -> pd.DataFrame | None:
    true_candidates = list(data_dir.glob("*True*.csv")) + list(data_dir.glob("*true*.csv")) + list(data_dir.glob("*REAL*.csv"))
    fake_candidates = list(data_dir.glob("*Fake*.csv")) + list(data_dir.glob("*fake*.csv")) + list(data_dir.glob("*FAKE*.csv"))
    if not true_candidates or not fake_candidates:
        return None

    true_frame = _standardize_frame(pd.read_csv(true_candidates[0]), label=0, source_name=true_candidates[0].name)
    fake_frame = _standardize_frame(pd.read_csv(fake_candidates[0]), label=1, source_name=fake_candidates[0].name)
    return pd.concat([true_frame, fake_frame], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def load_news_dataset(
    data_dir: str | Path = "data/raw",
    sample_path: str | Path = "data/sample_news.csv",
    use_sample_if_missing: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    sample_path = Path(sample_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    paired = _load_true_fake_pair(data_dir)
    if paired is not None:
        return paired

    csv_files = sorted(path for path in data_dir.glob("*.csv") if path.is_file())
    if csv_files:
        return _standardize_frame(pd.read_csv(csv_files[0]), source_name=csv_files[0].name)

    if use_sample_if_missing and sample_path.exists():
        return _standardize_frame(pd.read_csv(sample_path), source_name=sample_path.name)

    raise FileNotFoundError(
        "No dataset found. Add Kaggle True.csv and Fake.csv, or one labelled CSV, under data/raw/. "
        "For a smoke test only, run with --use-sample-if-missing."
    )


def dataset_summary(frame: pd.DataFrame) -> dict[str, object]:
    counts = frame["label_name"].value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "class_distribution": {str(key): int(value) for key, value in counts.items()},
        "average_text_length": float(frame["combined_text"].str.len().mean()),
    }
