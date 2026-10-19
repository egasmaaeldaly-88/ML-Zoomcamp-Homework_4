# ML-Zoomcamp-Homework_4


Solution notebook for **Homework 4** of the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (2026 cohort), covering model evaluation metrics for a binary classifier.

## Task

Predict whether a lead **converts** (signs up to the platform) using the 2026 lead-scoring dataset (`course_lead_scoring_2026.csv`, 5,000 rows, target: `converted`).

## Data preparation

- Missing categorical values filled with `'NA'`, missing numerical values filled with `0.0`.
- Split with the exact calls from the homework:
  - `train_test_split(df, test_size=0.2, random_state=1)` gives `df_full_train` (4,000 rows) and `df_test` (1,000 rows).
  - `train_test_split(df_full_train, test_size=0.25, random_state=1)` gives `df_train` (3,000 rows) and `df_val` (1,000 rows).

## What the notebook covers

| Question | Topic |
|---|---|
| Q1 | ROC AUC as feature importance for the numerical features (inverting a feature when its AUC is below 0.5) |
| Q2 | Logistic regression with `DictVectorizer` one-hot encoding and AUC on the validation set |
| Q3 | Precision and recall over thresholds 0.0 to 1.0 (step 0.01) and the point where the curves intersect |
| Q4 | F1 score over the same thresholds and the threshold that maximizes it |
| Q5 | 5-fold cross-validation (`KFold`) on `df_full_train` and the standard deviation of the AUC scores |
| Q6 | Hyperparameter tuning of `C` with 5-fold cross-validation (mean and std of AUC) |

Model used throughout:

```python
LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
```

## Results

Fill in after the final run.

| Question | Answer |
|---|---|
| Q1 | |
| Q2 | |
| Q3 | |
| Q4 | |
| Q5 | |
| Q6 | |

## How to run

1. Download the dataset:
   `https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv`
2. Install the requirements:
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```
3. Open `homework4.ipynb` and run all cells from top to bottom (Kernel, then Restart & Run All).

## Notes

- All model selection and evaluation (Q2 to Q4) is done on the **validation** set. The test set is not used for these questions.
- Cross-validation (Q5, Q6) is run on `df_full_train`.
