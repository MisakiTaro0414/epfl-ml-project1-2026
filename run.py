"""Generate submission.csv using saved, fully trained NumPy model artifacts."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data_utils import load_test
from helpers import create_csv_submission
from models import sigmoid
from preprocessing import Preprocessor


def predict(data_dir, artifact_dir, output):
    """Validate feature order and model files; preserve every test ID in order."""
    artifact_dir, output = Path(artifact_dir), Path(output)
    config = json.loads((artifact_dir / "config.json").read_text())
    for filename, key in (
        ("weights.npz", "weights_sha256"),
        ("preprocessor.json", "preprocessor_sha256"),
    ):
        digest = hashlib.sha256((artifact_dir / filename).read_bytes()).hexdigest()
        if digest != config[key]:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
    processor = Preprocessor.load(artifact_dir / "preprocessor.json")
    with np.load(artifact_dir / "weights.npz", allow_pickle=False) as archive:
        weights = archive["weights"]
    x, ids, names = load_test(data_dir)
    if names != processor.input_names:
        raise ValueError("Test feature names/order differ from the fitted model")
    if weights.shape != (len(processor.feature_names),):
        raise ValueError("Model and preprocessing dimensions do not match")
    predictions = np.empty(len(ids), dtype=np.int8)
    for start in range(0, len(x), 4096):
        stop = min(start + 4096, len(x))
        tx = processor.transform(x[start:stop])
        scores = tx @ (
            weights.astype(np.float32) if config["kind"] == "logistic" else weights
        )
        if config["kind"] == "logistic":
            scores = sigmoid(scores)
        predictions[start:stop] = np.where(scores >= config["threshold"], 1, -1)
    if len(ids) != len(predictions) or not np.isin(predictions, [-1, 1]).all():
        raise ValueError("Invalid submission predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(ids, predictions, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        f"Wrote {len(ids)} predictions to {output}; positives={int((predictions == 1).sum())}"
    )
    print(f"SHA256 {digest}")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--artifacts", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    predict(args.data_dir, args.artifacts, args.output)
