# Task III: Multi-Stage Ensemble Framework for Fabricated Digital Narratives

This folder contains the Task III implementation package for WQF7007 NLP: dataset loading, text preprocessing, baseline models, stacking ensemble training, Streamlit demonstration app, evaluation outputs, and stakeholder testing materials.

## Dataset

Place the Kaggle fake-news CSV files in:

```text
data/raw/
```

Supported formats:

- `True.csv` and `Fake.csv`
- one labelled CSV with a text column (`text`, `content`, `article`, `body`, or similar) and a label column (`label`, `class`, `target`, `type`, or `is_fake`)

The included `data/sample_news.csv` is only for smoke testing and classroom demo rehearsal. It is not a substitute for the full Kaggle dataset.

Current formal dataset loaded:

- `data/raw/WELFake_Dataset.csv`
- Raw rows: 72,134
- Cleaned training rows: 63,355

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train

With the full Kaggle dataset:

```bash
python train_model.py
```

Smoke test without Kaggle CSV:

```bash
python train_model.py --use-sample-if-missing
```

Outputs:

- `models/fake_news_stacking_pipeline.joblib`
- `reports/model_comparison.csv`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `reports/metrics_summary.json`
- `reports/training_summary.md`

## Run Streamlit Demo

```bash
streamlit run app.py
```

On Windows, the included helper can be used from PowerShell:

```powershell
.\run_streamlit.ps1
```

The app supports:

- direct news text input
- fake/true prediction
- class probabilities
- high-signal TF-IDF terms from the input
- model performance review
- stakeholder testing protocol

## Task III Coverage

- Preprocessing: URL/HTML removal, text normalization, duplicate removal, TF-IDF vectorization
- Application technique: supervised ML, baseline classifiers, stacking ensemble
- Architecture: input data, preprocessing, vectorization, base learners, meta learner, Streamlit interface
- Development: dataset loader, training pipeline, saved model artifacts, app prototype
- Evaluation: train/test split, baseline comparison, precision, recall, F1, confusion matrix
- Stakeholder testing: reusable testing template plus sample demo log
