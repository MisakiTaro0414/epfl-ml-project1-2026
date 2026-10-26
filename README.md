# EPFL CS-433 — Project 1 (Fall 2026)

**Team:** Ayhan Suleymanzade, Jasna Ilieva, Hayden Richard Stec.

[Open in Colab](https://colab.research.google.com/github/MisakiTaro0414/epfl-ml-project1-2026/blob/main/run.ipynb)

NumPy implementations of the six required algorithms, an executed classification
pipeline, saved model weights, reproducible predictions, and a two-page LaTeX report.
The prediction target is existing self-reported myocardial infarction/coronary heart
disease in the supplied BRFSS survey, not prospectively observed future disease.

## Run the saved model

Python 3.9+ and NumPy 1.23.1+ are sufficient for data loading, training, and prediction.
The recorded run used Python 3.12 and NumPy 2.5.4 on CPU. No pandas, scikit-learn,
SciPy, or deep-learning libraries are used by the project.

Compatibility is verified with the course grader's Python 3.9 and NumPy 1.23.1;
inference produces the same prediction CSV bytes as the recorded training environment.

```bash
python -m pip install -r requirements.txt
python download_data.py
python run.py
```

This downloads the official dataset and writes `submission.csv`, containing exactly
109,379 rows in the original test-ID order, with columns `Id,Prediction` and labels
`-1,+1`. Alternatively, place the three course CSVs in `data/dataset/`, or pass
`--data-dir /path/to/dataset` to each command. Dataset caches are created alongside
the CSVs, so this directory must be writable. `run.py` needs only `x_test.csv`.

The saved artifacts in `results/artifacts/` were trained on all 328,135 labeled rows.
The generated competition file is also included as `predictions/submission.csv`.
Inference validates the artifact checksums and the input feature order. The exact
recorded submission SHA256 is
`aa418240aa687a9b3b891998a3d077b3d0918825a0a8f2b223fe378b15e1cdf2`.
Repeated inference in the recorded environment produced identical bytes. Hardware,
BLAS, and NumPy differences can slightly affect floating-point retraining results;
use the committed artifacts to reproduce the submitted model.

## Reproduce the experiments

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python train.py
python run.py
```

Training ran in about 115 seconds after CSV caching on the recorded machine;
other CPUs will vary. Allow approximately 4 GB of RAM and 2 GB of disk space.
The default seed is 433 and the optimizer budget is 150 iterations per fit.
`--output-dir` can preserve the supplied results when running another experiment.

`train.py` creates a stratified 60/20/20 split: 196,881 training rows, 65,627 tuning
rows, and 65,627 holdout rows. Preprocessing is fitted only on training rows.
Hyperparameters and decision thresholds are chosen only on the tuning split.
The configuration is saved to `selection_before_holdout.json` before any holdout
predictions are computed. All candidate comparisons on the holdout are descriptive;
the final configuration is not changed after seeing them. The selected configuration
is then fitted on all labeled data, retaining its tuning-set threshold.

This is one fixed holdout experiment, not k-fold cross-validation. The split IDs,
per-row holdout scores, complete optimizer traces, source-file hashes and metrics
are written under `results/`. Raw data and per-row intermediate outputs are not
tracked in Git. The split and all outputs can be regenerated from the seed.

## Data preparation

- Source: 328,135 training and 109,379 test rows, with 321 raw predictors.
  Positive prevalence is 8.8302%; approximately 44.79% of feature cells are missing.
- Drop identifiers `SEQNO`, `_PSU`, `IDATE`, direct target columns if present, and
  `HAREHAB1`. The latter asks about rehabilitation after a heart attack; its survey
  skip logic depends on a target-defining answer. It is excluded before encoding.
- Drop columns with more than 95% missing training entries and constant columns.
  An otherwise constant column can still contribute a missingness indicator.
- Treat non-finite entries as missing. Do not globally recode values such as 7, 9,
  77 or 99: their meanings differ by survey question. This leaves a limitation for
  high-cardinality numeric columns containing question-specific sentinel codes.
- Numeric features use training medians, clipping at observed 0.5th/99.5th
  percentiles, then training means and standard deviations. Add missing indicators.
- Engineered features one-hot encode columns with at most 20 observed values,
  use the most frequent category as reference, group categories observed fewer
  than 20 times with unseen values, and add standardized squared continuous terms.
  The categorical missing case has a separate indicator if present during fitting.
- The development feature maps contain 414 numeric or 1,136 engineered columns,
  including the intercept. Refitting on all data produces 1,156 engineered columns.
  Stored preprocessing statistics always travel with the matching weights.

## Models and results

`implementations.py` provides MSE GD, batch-size-one MSE SGD, least squares,
ridge regression, logistic GD and regularized logistic GD. MSE is divided by two;
all functions return final weights and the unregularized loss. The required
regularized functions penalize every supplied weight, including the intercept.
Normal equations use a pseudoinverse fallback for singular systems, never `lstsq`.

The experiment compares least squares and ridge (`lambda=0.001`) on numeric
features, plus logistic models with each of `lambda=0.00001,0.0001,0.001` under
both feature representations.
The experiment's logistic optimizer is a separate NumPy implementation of L-BFGS,
with ten history pairs and Armijo backtracking. Its objective is mean binary
cross-entropy plus `lambda * ||w[1:]||²`, leaving the intercept unpenalized.
Each candidate's threshold maximizes tuning F1, considering tied scores together.

The selected model uses engineered features, `lambda=0.00001`, and threshold
`0.21064792573451996`. These are **local holdout results**, not AIcrowd scores:

| Model / threshold | F1 | Accuracy | Precision | Recall | AUROC |
|---|---:|---:|---:|---:|---:|
| Always negative | 0.0000 | 0.9117 | 0.0000 | 0.0000 | 0.5000 |
| Least squares, tuned threshold | 0.4133 | 0.8621 | 0.3310 | 0.5501 | 0.8503 |
| Ridge, tuned threshold | 0.4145 | 0.8603 | 0.3289 | 0.5601 | 0.8497 |
| Selected logistic, threshold 0.5 | 0.2540 | 0.9142 | 0.5477 | 0.1653 | 0.8578 |
| Selected logistic, tuned threshold | **0.4214** | 0.8736 | 0.3536 | 0.5213 | 0.8578 |

The selected holdout F1 has a 95% conditional bootstrap interval of
`[0.4113, 0.4319]` from 2,000 multinomial resamples of its confusion counts. This
interval does not include training/selection uncertainty or survey-design effects.
The majority classifier illustrates why accuracy alone is misleading here. The
tuned threshold improves F1 while lowering accuracy. Feature engineering gives a
modest improvement; the held-out comparisons do not establish statistical
significance. See `results/metrics.json` for every candidate and `report/report.pdf`
for the scientific discussion.

The final optimizer reached its 150-iteration budget with maximum gradient
component about `1.58e-4`; exact numerical convergence is not claimed. Data are
cross-sectional, self-reported, and not reweighted using the complex survey design.
This model is a course classification exercise, not a clinical decision system.

## Validation and report

See [the guidelines audit](GUIDELINES_AUDIT.md) for requirement-by-requirement
evidence and the outstanding AIcrowd and course-portal submission steps.

```bash
python -m unittest discover -s tests -v
python run.py --output submission_reproduced.csv
cmp predictions/submission.csv submission_reproduced.csv
python report/build_report.py
```

Validation passed 28 independent tests and all 27 applicable official course
checks in a clean submission directory using Python 3.9.25, NumPy 1.23.1,
pytest 7.1.2, Black 22.6.0, and GitPython 3.1.18. The remaining official check
only tests the remote GitHub URL format; it is excluded when testing a local path.
The companion notebook was executed locally without cell errors.

The report builder needs Matplotlib and a working LaTeX installation. It regenerates
the evaluation diagram and comparison figures from the recorded experiment metrics.
Development-only notebook,
formatting, plotting and course-test tools are listed in `requirements-dev.txt`;
they are not dependencies of prediction or training.

The course tests are kept outside the deliverable. To run them against a clean
checkout (without an in-repository virtual environment), use the official
`grading_tests/` instructions and omit only the GitHub URL-format check when
passing a local path. For final submission, provide a GitHub URL ending in
`/tree/<commit-sha>`, as requested by that test.

## Files

| File | Purpose |
|---|---|
| `implementations.py` | Six required functions |
| `data_utils.py`, `preprocessing.py` | Validated CSV loading and fitted feature maps |
| `models.py`, `train.py` | Optimization, model selection, holdout evaluation, final fit |
| `run.py`, `run.ipynb` | Reproduce predictions from committed artifacts |
| `download_data.py` | Download and verify official data |
| `helpers.py` | Original course CSV helper, with formatting only |
| `results/metrics.json`, `results/artifacts/` | Executed results and final saved model |
| `tests/` | Independent numerical and preprocessing tests |
| `report/` | LaTeX source, result-driven builder and PDF |

## Provenance and final submission

This repository was prepared with AI assistance and tested locally. The original
team Colab link required Google authentication and was not available to inspect;
its code has not been merged or overwritten. The companion notebook in this
repository is a separate reproducible entry point. Execution recorded here took
place on the local CPU environment, not an authenticated Google Colab runtime.
Team members should review the implementation and report before submitting.
No AIcrowd prediction upload or final course submission
has been made.

Primary sources:

- [Official assignment and starter files](https://github.com/epfml/ML_course/tree/main/projects/project1)
- [Official public tests](https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests)
- [CDC 2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf),
  printed pages 73 (`HAREHAB1`) and 106 (`_MICHD`).
- [AIcrowd leaderboard](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/leaderboards)
  displays F1 first and accuracy second. Its rules page describes accuracy, so both
  metrics are reported here and tuning targets the leaderboard's F1 metric.
