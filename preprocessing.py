"""Train-fitted, serializable feature processing using only NumPy.

Raw survey codes are deliberately not interpreted as generic missing-value
sentinels: a value such as 7 or 99 can be a legitimate response in some columns.
Only non-finite entries are missing. Derived target columns, identifiers, and a
heart-attack follow-up question are excluded conservatively.
"""

import json
from pathlib import Path

import numpy as np

EXCLUDED_COLUMNS = {
    "Id",
    "ID",
    "SEQNO",
    "_PSU",
    "IDATE",
    "CVDINFR4",
    "CVDCRHD4",
    "_MICHD",
    "HAREHAB1",
}


class Preprocessor:
    """Fit on training rows; transform any split without relearning statistics.

    ``numeric`` applies median imputation, 0.5/99.5 percentile clipping and
    standardization. ``engineered`` additionally one-hot encodes columns with at
    most 20 observed values and adds squared continuous features. Categories
    observed fewer than 20 times are grouped with unknown categories. The most
    frequent retained category is the reference. Missing indicators are included
    whenever training has both observed and missing values. Columns with over
    95% missing data are omitted. Every output has a leading intercept column.
    """

    def __init__(self, mode="engineered"):
        if mode not in {"numeric", "engineered"}:
            raise ValueError("mode must be numeric or engineered")
        self.mode = mode
        self.columns = []
        self.feature_names = []
        self.input_names = []
        self.dropped = {}
        self.fitted = False

    def fit(self, x, names):
        """Learn all selection, imputation, scaling and category statistics."""
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != len(names):
            raise ValueError("Expected nonempty 2D data and matching feature names")
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique")
        self.columns = []
        self.feature_names = ["intercept"]
        self.input_names = list(names)
        self.dropped = {}
        for j, name in enumerate(names):
            if name in EXCLUDED_COLUMNS:
                self.dropped[name] = "identifier or target-related"
                continue
            raw = x[:, j].astype(np.float64)
            valid = np.isfinite(raw)
            missing_fraction = float(1.0 - valid.mean())
            if missing_fraction > 0.95:
                self.dropped[name] = "over 95% missing"
                continue
            observed = raw[valid]
            values, counts = np.unique(observed, return_counts=True)
            missing = bool(np.any(~valid))
            if len(values) == 1:
                if missing:
                    self.columns.append({"index": j, "kind": "missing", "name": name})
                    self.feature_names.append(name + "__missing")
                else:
                    self.dropped[name] = "constant"
                continue
            spec = {"index": j, "name": name, "missing": missing}
            if self.mode == "engineered" and len(values) <= 20:
                frequent = values[counts >= 20]
                if not len(frequent):
                    frequent = values[[int(np.argmax(counts))]]
                reference = float(values[int(np.argmax(counts))])
                cats = [float(v) for v in frequent if v != reference]
                spec.update(
                    kind="categorical",
                    categories=cats,
                    known=[float(v) for v in frequent],
                    reference=reference,
                )
                self.columns.append(spec)
                self.feature_names.extend(name + "==" + str(v) for v in cats)
                self.feature_names.append(name + "__other")
            else:
                median = float(np.median(observed))
                lower, upper = np.quantile(observed, [0.005, 0.995])
                if lower == upper:
                    lower, upper = observed.min(), observed.max()
                clean = np.clip(np.where(valid, raw, median), lower, upper)
                center = float(clean.mean())
                scale = float(clean.std())
                if scale < 1e-12:
                    self.dropped[name] = "constant after clipping"
                    continue
                z = (clean - center) / scale
                square_center = float(np.mean(z * z))
                square_scale = float(np.std(z * z))
                squared = self.mode == "engineered" and square_scale > 1e-12
                spec.update(
                    kind="numeric",
                    median=median,
                    lower=float(lower),
                    upper=float(upper),
                    center=center,
                    scale=scale,
                    squared=squared,
                    square_center=square_center,
                    square_scale=square_scale,
                )
                self.columns.append(spec)
                self.feature_names.append(name)
                if squared:
                    self.feature_names.append(name + "__squared")
            if missing:
                self.feature_names.append(name + "__missing")
        self.fitted = True
        return self

    def transform(self, x):
        """Return finite float32 features using only stored training statistics."""
        if not self.fitted:
            raise ValueError("Call fit before transform")
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != len(self.input_names):
            raise ValueError("Feature matrix does not match fitted input schema")
        output = np.empty((len(x), len(self.feature_names)), dtype=np.float32)
        output[:, 0] = 1.0
        k = 1
        for spec in self.columns:
            raw = x[:, spec["index"]].astype(np.float64)
            valid = np.isfinite(raw)
            if spec["kind"] == "missing":
                output[:, k] = ~valid
                k += 1
                continue
            if spec["kind"] == "categorical":
                for value in spec["categories"]:
                    output[:, k] = valid & (raw == value)
                    k += 1
                output[:, k] = valid & ~np.isin(raw, spec["known"])
                k += 1
            else:
                clean = np.clip(
                    np.where(valid, raw, spec["median"]), spec["lower"], spec["upper"]
                )
                z = (clean - spec["center"]) / spec["scale"]
                output[:, k] = z
                k += 1
                if spec["squared"]:
                    output[:, k] = (z * z - spec["square_center"]) / spec[
                        "square_scale"
                    ]
                    k += 1
            if spec["missing"]:
                output[:, k] = ~valid
                k += 1
        return output

    def save(self, path):
        """Save portable JSON statistics; no pickled executable objects."""
        if not self.fitted:
            raise ValueError("Cannot save an unfitted preprocessor")
        Path(path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "mode": self.mode,
                    "columns": self.columns,
                    "feature_names": self.feature_names,
                    "input_names": self.input_names,
                    "dropped": self.dropped,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path):
        """Restore exactly the feature mapping and training statistics."""
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported preprocessor format")
        obj = cls(data["mode"])
        for attr in ["columns", "feature_names", "input_names", "dropped"]:
            setattr(obj, attr, data[attr])
        obj.fitted = True
        return obj
