"""Read the supplied CSV files with NumPy and the standard library only.

Feature arrays are cached beside the source files as float32 .npy files. Cache
metadata contains the source size and nanosecond modification time; changing a
CSV invalidates its cache. Labels are always read and checked against row IDs.
"""

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _signature(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_features(path):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    if not header or header[0] != "Id":
        raise ValueError(f"{path.name}: first column must be Id")
    names = header[1:]
    if not names or len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError(f"{path.name}: empty or duplicate feature names")
    signature = _signature(path)
    cache_dir = path.parent / ".numpy_cache"
    cache_dir.mkdir(exist_ok=True)
    prefix = cache_dir / path.stem
    meta_path = prefix.with_suffix(".json")
    x_path = prefix.with_suffix(".npy")
    ids_path = prefix.with_suffix(".ids.npy")
    if meta_path.exists() and x_path.exists() and ids_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta["signature"] == signature and meta["names"] == names:
                x = np.load(x_path, mmap_mode="r", allow_pickle=False)
                ids = np.load(ids_path, allow_pickle=False)
                if x.shape == (len(ids), len(names)) and x.dtype == np.float32:
                    return x, ids, names
        except (OSError, ValueError, KeyError):
            pass
    # Count rows without retaining the CSV text. The temporary memmap bounds RAM.
    with path.open(newline="") as handle:
        n_rows = sum(1 for _ in handle) - 1
    if n_rows <= 0:
        raise ValueError(f"{path.name}: no data rows")
    temporary = tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npy", delete=False)
    temp_path = Path(temporary.name)
    temporary.close()
    try:
        x = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.float32, shape=(n_rows, len(names))
        )
        ids = np.empty(n_rows, dtype=np.int64)
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for i, row in enumerate(reader):
                if len(row) != len(header):
                    raise ValueError(f"{path.name}: wrong column count at row {i + 2}")
                try:
                    ids[i] = int(row[0])
                    x[i] = [float(v) if v.strip() else np.nan for v in row[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"{path.name}: invalid number at row {i + 2}"
                    ) from exc
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{path.name}: duplicate row IDs")
        x.flush()
        del x
        os.replace(temp_path, x_path)
        np.save(ids_path, ids, allow_pickle=False)
        meta_path.write_text(json.dumps({"signature": signature, "names": names}))
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return np.load(x_path, mmap_mode="r", allow_pickle=False), ids, names


def load_training(data_dir):
    """Return (features float32, labels 0/1, row IDs, feature names)."""
    data_dir = Path(data_dir)
    x, ids, names = _read_features(data_dir / "x_train.csv")
    labels = []
    label_ids = []
    with (data_dir / "y_train.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != 2 or header[0] != "Id":
            raise ValueError("y_train.csv must contain Id and one label column")
        for row in reader:
            if len(row) != 2:
                raise ValueError("y_train.csv must have two columns in every row")
            label_ids.append(int(row[0]))
            labels.append(int(row[1]))
    label_ids = np.asarray(label_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    if not np.array_equal(label_ids, ids):
        raise ValueError("Training label IDs do not match feature IDs in order")
    if not np.all(np.isin(labels, [-1, 1])):
        raise ValueError("Training labels must be -1 or +1")
    return x, (labels == 1).astype(np.int8), ids, names


def load_test(data_dir):
    """Return (test features float32, row IDs, feature names)."""
    return _read_features(Path(data_dir) / "x_test.csv")
