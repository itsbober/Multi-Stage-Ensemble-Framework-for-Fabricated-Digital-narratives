# Task III Model Evaluation Summary

## Dataset
- Rows used: 63355
- Class distribution: {'True': 34521, 'Fake': 28834}
- Average cleaned text length: 3309.3 characters

## Model Comparison
```text
                  model  accuracy  precision   recall       f1
      Stacking Ensemble  0.961092   0.954655 0.960118 0.957379
    Logistic Regression  0.957857   0.949185 0.958731 0.953934
          Random Forest  0.945782   0.944367 0.936015 0.940172
Multinomial Naive Bayes  0.862600   0.834942 0.870123 0.852169
```

## Selected Model
- Final application model: Stacking ensemble over Logistic Regression, Multinomial Naive Bayes, and Random Forest.
- Test F1: 0.9574

## Interpretation
- Logistic Regression handles sparse TF-IDF features and gives a strong linear baseline.
- Multinomial Naive Bayes provides a fast probabilistic text baseline.
- Random Forest adds non-linear interactions.
- Stacking learns how to combine the base models instead of relying on a fixed majority vote.