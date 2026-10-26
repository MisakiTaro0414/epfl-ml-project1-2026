"""Generate the project report from recorded experiments, then compile LaTeX.

Run from the project root with ``python report/build_report.py``. Results must
exist first: this script never substitutes made-up metrics for unfinished runs.
Figures use Matplotlib for visualization; PDF compilation requires pdflatex.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

from build_figures import generate_figures

HERE = Path(__file__).resolve().parent


def scientific(value):
    """Format positive powers of ten for a compact LaTeX table."""
    if not value:
        return "--"
    exponent = int(round(math.log10(value)))
    return rf"$10^{{{exponent}}}$"


def build(metrics_path, output_dir, compile_pdf=True):
    """Render only completed, recorded results; optionally compile twice."""
    result = json.loads(Path(metrics_path).read_text())
    experiments = result["experiments"]
    selected = result["selected"]
    chosen = next(row for row in experiments if row["name"] == selected["name"])
    holdout = result["selected_holdout"]
    default = result["selected_default_threshold_holdout"]
    majority = result["majority_holdout"]
    numeric = max(
        (
            row
            for row in experiments
            if row["mode"] == "numeric" and row["kind"] == "logistic"
        ),
        key=lambda row: row["tuning"]["f1"],
    )
    engineered = max(
        (row for row in experiments if row["mode"] == "engineered"),
        key=lambda row: row["tuning"]["f1"],
    )
    same_lambda = next(
        row
        for row in experiments
        if row["mode"] == "numeric"
        and row["kind"] == "logistic"
        and row["lambda"] == engineered["lambda"]
    )
    label = {"least_squares": "Least squares", "ridge": "Ridge", "logistic": "Logistic"}
    rows = [
        rf"Majority & -- & -- & {majority['f1']:.4f} & {majority['accuracy']:.4f} \\"
    ]
    for row in experiments:
        method = ("N " if row["mode"] == "numeric" else "E ") + label[row["kind"]]
        if row["name"] == selected["name"]:
            method = r"\textbf{" + method + "}"
        rows.append(
            rf"{method} & {scientific(row['lambda'])} & {row['tuning']['f1']:.4f}"
            rf" & {row['holdout']['f1']:.4f} & {row['holdout']['accuracy']:.4f} \\"
        )
    data = result["data"]
    split = result["split_sizes"]
    summary_rows = [
        rf"Always negative & -- & {majority['f1']:.4f} & {majority['accuracy']:.4f} \\",
    ]
    for row in experiments:
        if row["kind"] in {"least_squares", "ridge"} or row["name"] == selected["name"]:
            name = label[row["kind"]]
            if row["name"] == selected["name"]:
                name = r"\textbf{Selected logistic}"
            summary_rows.append(
                rf"{name} & {row['tuning']['f1']:.4f}"
                rf" & {row['holdout']['f1']:.4f} & {row['holdout']['accuracy']:.4f} \\",
            )
    tokens = {
        "NROWS": f"{data['rows']:,}",
        "NPOS": f"{data['positives']:,}",
        "PREVALENCE": f"{100 * data['prevalence']:.2f}",
        "NTRAIN": f"{split['train']:,}",
        "NTUNE": f"{split['tuning']:,}",
        "NHOLD": f"{split['holdout']:,}",
        "NFEATURES": str(numeric["features"]),
        "EFEATURES": str(engineered["features"]),
        "MAXITERS": str(result["max_iters"]),
        "SEED": str(result["seed"]),
        "LAMBDA": scientific(selected["lambda"]),
        "THRESHOLD": f"{selected['threshold']:.5f}",
        "F1": f"{holdout['f1']:.4f}",
        "ACCURACY": f"{holdout['accuracy']:.4f}",
        "AUC": f"{holdout['auc']:.4f}",
        "PRECISION": f"{holdout['precision']:.4f}",
        "RECALL": f"{holdout['recall']:.4f}",
        "CILOW": f"{result['selected_holdout_f1_ci95'][0]:.4f}",
        "CIHIGH": f"{result['selected_holdout_f1_ci95'][1]:.4f}",
        "TP": f"{holdout['tp']:,}",
        "FP": f"{holdout['fp']:,}",
        "FN": f"{holdout['fn']:,}",
        "TN": f"{holdout['tn']:,}",
        "DEFAULTF1": f"{default['f1']:.4f}",
        "DEFAULTACC": f"{default['accuracy']:.4f}",
        "DEFAULTPREC": f"{default['precision']:.4f}",
        "DEFAULTREC": f"{default['recall']:.4f}",
        "THRESHOLDGAIN": f"{holdout['f1'] - default['f1']:+.4f}",
        "FEATUREGAIN": f"{engineered['holdout']['f1'] - same_lambda['holdout']['f1']:+.4f}",
        "FEATURELAMBDA": scientific(engineered["lambda"]),
        "NUMF1": f"{numeric['holdout']['f1']:.4f}",
        "ENF1": f"{engineered['holdout']['f1']:.4f}",
        "NUMPY": result["numpy_version"],
        "SELECTEDMODE": selected["mode"],
        "SELECTEDKIND": label[selected["kind"]].lower(),
        "ITERATIONS": str(chosen["optimizer"].get("iterations", "not iterative")),
        "FINALFEATURES": str(result["final_features"]),
        "TABLE": "\n".join(rows),
        "SUMMARYTABLE": "\n".join(summary_rows),
        "MATCHEDF1": f"{same_lambda['holdout']['f1']:.4f}",
        "MAJACCPCT": f"{100 * majority['accuracy']:.2f}",
        "DEFAULTRECPCT": f"{100 * default['recall']:.1f}",
        "RECPCT": f"{100 * holdout['recall']:.1f}",
        "DEFAULTPRECPCT": f"{100 * default['precision']:.1f}",
        "PRECPCT": f"{100 * holdout['precision']:.1f}",
        "DEFAULTACCPCT": f"{100 * default['accuracy']:.2f}",
        "ACCPCT": f"{100 * holdout['accuracy']:.2f}",
    }
    document = (HERE / "report_template.tex").read_text()
    for key, value in tokens.items():
        document = document.replace("@@" + key + "@@", value)
    if "@@" in document:
        raise ValueError("Unresolved report substitution")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_figures(metrics_path, output_dir / "figures")
    tex = output_dir / "report.tex"
    tex.write_text(document)
    if compile_pdf:
        executable = shutil.which("pdflatex")
        if executable is None:
            candidate = Path.home() / "bin" / "pdflatex"
            if candidate.exists():
                executable = str(candidate)
        if executable is None:
            raise FileNotFoundError(
                "pdflatex is required; report.tex has been generated"
            )
        for _ in range(2):
            process = subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode:
                raise RuntimeError(process.stdout[-6000:] + process.stderr[-2000:])
        print(output_dir / "report.pdf")
    return tex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=HERE.parent / "results" / "metrics.json"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--no-compile", action="store_true")
    arguments = parser.parse_args()
    build(arguments.metrics, arguments.output_dir, not arguments.no_compile)
