"""Reproduce model selection, an untouched holdout evaluation, and final fit."""

import os

# Set before importing NumPy; avoid oversubscribing a shared machine.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from data_utils import load_training
from implementations import least_squares, ridge_regression
from models import best_f1_threshold, fit_logistic, metrics, sigmoid, stratified_split
from preprocessing import Preprocessor


def sha256(path):
    """Stream a file digest without loading the CSV into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(data_dir, output_dir, seed=433, max_iters=150):
    """Tune solely on 20%; evaluate fixed candidates on a separate 20%."""
    start = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    x, y, ids, names = load_training(data_dir)
    train_idx, tune_idx, holdout_idx = stratified_split(y, seed)
    np.savez_compressed(
        output_dir / "split_ids.npz",
        train=ids[train_idx],
        tuning=ids[tune_idx],
        holdout=ids[holdout_idx],
    )
    results = {
        "seed": seed,
        "numpy_version": np.__version__,
        "max_iters": max_iters,
        "data": {
            "rows": len(y),
            "raw_features": len(names),
            "positives": int(y.sum()),
            "prevalence": float(y.mean()),
        },
        "split_sizes": {
            "train": len(train_idx),
            "tuning": len(tune_idx),
            "holdout": len(holdout_idx),
        },
        "split_positives": {
            k: int(y[idx].sum())
            for k, idx in zip(
                ("train", "tuning", "holdout"), (train_idx, tune_idx, holdout_idx)
            )
        },
        "source_sha256": {
            name: sha256(Path(data_dir) / name)
            for name in ("x_train.csv", "y_train.csv")
        },
        "experiments": [],
    }
    candidates = []
    for mode in ("numeric", "engineered"):
        print(f"Preparing {mode} features", flush=True)
        processor = Preprocessor(mode).fit(x[train_idx], names)
        tx = processor.transform(x[train_idx])
        vx = processor.transform(x[tune_idx])
        processor.save(output_dir / f"{mode}_development_preprocessor.json")
        specifications = [("logistic", value) for value in (1e-5, 1e-4, 1e-3)]
        if mode == "numeric":
            specifications = [("least_squares", 0.0), ("ridge", 1e-3)] + specifications
        for kind, regularization in specifications:
            label = f"{mode}_{kind}_{regularization:g}"
            print(f"Fitting {label} ({tx.shape[1]} columns)", flush=True)
            before = time.monotonic()
            if kind == "logistic":
                w, optimizer = fit_logistic(tx, y[train_idx], regularization, max_iters)
                scores = sigmoid(vx @ w.astype(np.float32))
            else:
                function = (
                    least_squares if kind == "least_squares" else ridge_regression
                )
                args = () if kind == "least_squares" else (regularization,)
                w, loss = function(y[train_idx], tx, *args)
                scores = vx @ w
                optimizer = {"unpenalized_training_loss": float(loss)}
            threshold = best_f1_threshold(y[tune_idx], scores)
            entry = {
                "name": label,
                "mode": mode,
                "kind": kind,
                "lambda": regularization,
                "features": tx.shape[1],
                "tuning": metrics(y[tune_idx], scores, threshold),
                "optimizer": optimizer,
                "fit_seconds": time.monotonic() - before,
            }
            results["experiments"].append(entry)
            candidates.append((entry, processor, w))
            print(
                f"  tuning F1={entry['tuning']['f1']:.5f}, "
                f"accuracy={entry['tuning']['accuracy']:.5f}, "
                f"threshold={threshold:.5f}, {entry['fit_seconds']:.1f}s",
                flush=True,
            )
            (output_dir / "progress.json").write_text(json.dumps(results, indent=2))
        del tx, vx

    # Selection occurs before any holdout scores are calculated.
    selected = max(candidates, key=lambda candidate: candidate[0]["tuning"]["f1"])
    results["selected"] = {
        key: selected[0][key] for key in ("name", "mode", "kind", "lambda", "features")
    }
    results["selected"]["threshold"] = selected[0]["tuning"]["threshold"]
    (output_dir / "selection_before_holdout.json").write_text(
        json.dumps(results["selected"], indent=2)
    )
    print(f"Selection frozen: {selected[0]['name']}", flush=True)
    results["majority_holdout"] = metrics(
        y[holdout_idx], np.zeros(len(holdout_idx)), 0.5
    )
    current_mode, hx = None, None
    for entry, processor, w in candidates:
        if entry["mode"] != current_mode:
            hx = processor.transform(x[holdout_idx])
            current_mode = entry["mode"]
        scores = hx @ (w.astype(np.float32) if entry["kind"] == "logistic" else w)
        if entry["kind"] == "logistic":
            scores = sigmoid(scores)
        entry["holdout"] = metrics(y[holdout_idx], scores, entry["tuning"]["threshold"])
        if entry["name"] == selected[0]["name"]:
            results["selected_holdout"] = entry["holdout"]
            results["selected_default_threshold_holdout"] = metrics(
                y[holdout_idx], scores, 0.5
            )
            np.savez_compressed(
                output_dir / "holdout_predictions.npz",
                ids=ids[holdout_idx],
                y=y[holdout_idx],
                scores=scores,
            )
            counts = [entry["holdout"][k] for k in ("tp", "fp", "fn", "tn")]
            draws = np.random.default_rng(seed + 1).multinomial(
                len(holdout_idx), np.array(counts) / len(holdout_idx), size=2000
            )
            bootstrap_f1 = (
                2
                * draws[:, 0]
                / np.maximum(2 * draws[:, 0] + draws[:, 1] + draws[:, 2], 1)
            )
            results["selected_holdout_f1_ci95"] = np.quantile(
                bootstrap_f1, [0.025, 0.975]
            ).tolist()
    del hx, candidates

    print("Refitting the frozen configuration on all labeled rows", flush=True)
    final_processor = Preprocessor(results["selected"]["mode"]).fit(x, names)
    tx = final_processor.transform(x)
    kind = results["selected"]["kind"]
    regularization = results["selected"]["lambda"]
    if kind == "logistic":
        w, final_optimizer = fit_logistic(tx, y, regularization, max_iters)
    elif kind == "ridge":
        w, loss = ridge_regression(y, tx, regularization)
        final_optimizer = {"unpenalized_training_loss": float(loss)}
    else:
        w, loss = least_squares(y, tx)
        final_optimizer = {"unpenalized_training_loss": float(loss)}
    final_processor.save(artifacts / "preprocessor.json")
    np.savez_compressed(artifacts / "weights.npz", weights=w)
    config = {
        **results["selected"],
        "features": tx.shape[1],
        "seed": seed,
        "training_rows": len(y),
        "numpy_version": np.__version__,
        "label_mapping": {"negative": -1, "positive": 1},
        "weights_sha256": sha256(artifacts / "weights.npz"),
        "preprocessor_sha256": sha256(artifacts / "preprocessor.json"),
    }
    (artifacts / "config.json").write_text(json.dumps(config, indent=2))
    results["final_optimizer"] = final_optimizer
    results["final_features"] = tx.shape[1]
    results["elapsed_seconds"] = time.monotonic() - start
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "selected": results["selected"],
                "holdout": results["selected_holdout"],
                "seconds": results["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=433)
    parser.add_argument("--max-iters", type=int, default=150)
    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir, args.seed, args.max_iters)
