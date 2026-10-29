
# CS-433 Machine Learning — Project 1

**EPFL, Fall 2026**  
**Team:** Ayhan Suleymanzade, Jasna Ilieva, Hayden Richard Stec

This project uses the 2015 Behavioral Risk Factor Surveillance System (BRFSS) dataset to predict whether a respondent has reported coronary heart disease or a heart attack. It includes the six required machine learning implementations, our classification pipeline, and the [project report](report/report.pdf).

## Getting started

The project uses Python and NumPy, without external machine learning libraries.

```bash
pip install -r requirements.txt
python download_data.py
python run.py
```

`run.py` loads the saved model and generates `submission.csv` (`Id,Prediction`). Alternatively, download the dataset from [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1) and place `x_train.csv`, `y_train.csv`, and `x_test.csv` in `data/dataset/`.

To reproduce training and evaluation, run:

```bash
python train.py
```

## Methods

`implementations.py` contains the six required methods: MSE gradient descent and stochastic gradient descent, least squares, ridge regression, logistic regression, and regularized logistic regression.

For prediction, we remove identifiers and target-leaking features, plus columns with over 95% missing values or no variation. Numeric features are median-imputed, clipped and standardized; we also use missingness indicators, one-hot encodings for low-cardinality features, and squared numeric features. We compare least squares, ridge, and regularized logistic regression using a stratified **60/20/20 train/validation/holdout split**. Preprocessing is fitted on training data, and hyperparameters and thresholds are chosen on validation F1. The holdout is used for evaluation; no k-fold cross-validation is performed.

Our final model is regularized logistic regression with engineered features, trained with a NumPy L-BFGS optimizer (`λ = 10⁻⁵`, decision threshold `0.21065`). It was subsequently refitted on the full labeled training dataset for submission.

## Results

**AIcrowd competition leaderboard — submitted model**

| F1 score | Accuracy |
|---:|---:|
| **0.440** | **0.879** |

For comparison, the selected model achieved **0.4214 F1** and **0.8736 accuracy** on our local holdout split. Additional experimental results are available in [`results/metrics.json`](results/metrics.json) and the [report](report/report.pdf).

## Repository structure

| File or directory | Description |
|---|---|
| `implementations.py` | Six required machine learning methods |
| `preprocessing.py`, `data_utils.py` | Data loading and feature preparation |
| `models.py`, `train.py` | Model training and evaluation |
| `run.py`, `run.ipynb` | Generate predictions from the saved model |
| `results/` | Metrics and saved model |
| `report/` | LaTeX report and PDF |

[Assignment description](https://github.com/epfml/ML_course/tree/main/projects/project1)
