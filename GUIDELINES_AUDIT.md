# Project 1 guidelines audit

Checked on 26 October 2026 against the official Fall 2026 assignment and public
tests at course commit
[`36e71e36e4d1f7d18cbc63d38409815a24bdc0c7`](https://github.com/epfml/ML_course/tree/36e71e36e4d1f7d18cbc63d38409815a24bdc0c7/projects/project1).
The downloaded assignment, helper, grading instructions, tests, and environment
file match that commit. This audit covers this repository and its generated PDF
and predictions; the private original team Colab was not accessible.

## Repository checks

| Requirement | Evidence | Result |
|---|---|---|
| Six specified functions in `implementations.py` | Exact signatures, 1D vectors, final `(w, loss)` returns | Pass |
| Required loss definitions | MSE divided by two; mean logistic cross-entropy; returned losses exclude regularization | Pass |
| Correct algorithms | Full-batch GD, one-observation SGD, normal-equation least squares/ridge; no `lstsq` calls | Pass |
| Allowed libraries | Tracked Python and notebook code imports only standard-library modules, local modules, NumPy, and Matplotlib for visualization | Pass |
| Official data only | Official archive and all four CSV checksums verified; no external training data or pretrained external models | Pass |
| Local evaluation | Stratified 60/20/20 split; train-fitted preprocessing; model and threshold frozen before holdout evaluation | Pass |
| Reproducible prediction script | `run.py` loads committed model/preprocessing; `train.py` documents and reproduces training | Pass |
| Prediction format | 109,379 rows; `Id,Prediction`; original test IDs in order; labels only `-1,+1`; course submission writer used | Pass |
| Public code and documentation | Public GitHub repository, README, source, notebook, trained artifacts, and exact prediction CSV | Pass |
| Report format | LaTeX-generated PDF: exactly two pages including references, no appendix, all three team names | Pass |
| Report content | Methods, preprocessing, search values, split protocol, baselines, feature and threshold ablations, limitations, and reproducibility | Pass |
| Accuracy of claims | Published numbers match recorded experiments; local holdout and competition results are distinguished | Pass |

The required regularized functions penalize every supplied weight, including an
intercept. The separate predictive L-BFGS model exempts its intercept, as explicitly
described in the report. The assignment permits additional methods for prediction.

Development tools in `requirements-dev.txt` are not modeling dependencies. The
official grading environment itself uses pytest, Black, and GitPython. Notebook
hosting and PDF inspection tools are used only during development; no such imports
are needed in submitted notebook cells or modeling code. The optional Colab SDK
download helper was replaced with instructions for the Files pane.

## Executed checks

- Clean project snapshot: **28 independent numerical/preprocessing tests passed**.
- Official public suite on that snapshot: **27 applicable checks passed**; only
  the GitHub URL-format check is omitted when passing a local directory.
- Environment: Python **3.9.25**, NumPy **1.23.1**, pytest **7.1.2**, Black **22.6.0**,
  GitPython **3.1.18**, matching the versions specified by the grader where pinned
  (the grader specifies the Python 3.9 series).
- Inference in this environment reproduced the committed CSV byte for byte,
  matching inference in the original Python 3.12 / NumPy 2.5.4 environment.
- Notebook cells were rerun locally after removing the Colab SDK import.
- Report rebuilt successfully with no overfull boxes; the PDF remains two pages.

Prediction SHA256:
`aa418240aa687a9b3b891998a3d077b3d0918825a0a8f2b223fe378b15e1cdf2`.

The public tests are not a guarantee of full credit; code, methodology, and report
quality also receive human review. The repository discloses AI assistance in the
README. No additional AI-specific policy is established by this project PDF.

## Submission steps still outstanding

1. Upload `predictions/submission.csv` to the team's
   [AIcrowd competition account](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).
   The limit is five submissions per team per day. No upload or online score has
   been recorded by this workflow. The requirement to reproduce the team's best
   competition entry remains unverified until its submission history is checked.
2. Submit `report/report.pdf` and the public repository URL for a precise commit
   (`https://github.com/MisakiTaro0414/epfl-ml-project1-2026/tree/<commit-sha>`) to
   [the course portal](http://mlcourse.epfl.ch). Obtain the commit with
   `git rev-parse HEAD` after the final changes are pushed.
3. The assignment deadline is **29 October 2026 at 16:00**. If the final model's
   online performance differs from the team's best entry, report that difference
   as required by Step 4.

The original team Colab and any prior competition submissions have not been
incorporated or audited because they require access that was not available here.
