"""Checks for split leakage, missing values, schema integrity and persistence."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_utils import load_test, load_training
from preprocessing import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def test_frozen_statistics_and_serialization(self):
        x = np.column_stack([np.arange(100), np.tile([1, 2], 50)]).astype(float)
        x[0, 0] = np.nan
        p = Preprocessor("engineered").fit(x, ["continuous", "category"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processor.json"
            p.save(path)
            before = path.read_text()
            test = np.array([[1e12, 100], [np.nan, np.nan]])
            z = p.transform(test)
            self.assertTrue(np.isfinite(z).all())
            self.assertTrue(np.all(z[:, 0] == 1))
            np.testing.assert_array_equal(z, Preprocessor.load(path).transform(test))
            p.save(path)
            self.assertEqual(before, path.read_text())
            self.assertEqual(z[0, p.feature_names.index("category__other")], 1)
            self.assertEqual(z[1, p.feature_names.index("category__other")], 0)

    def test_missing_constant_and_leaking_columns(self):
        x = np.column_stack(
            [
                np.full(100, np.nan),
                np.ones(100),
                np.arange(100),
                np.arange(100),
                np.ones(100),
            ]
        )
        x[0, -1] = np.nan
        p = Preprocessor("numeric").fit(
            x, ["empty", "constant", "_MICHD", "SEQNO", "flag"]
        )
        self.assertEqual(p.feature_names, ["intercept", "flag__missing"])
        z = p.transform(x)
        self.assertEqual(z.shape, (100, 2))
        self.assertEqual(z[0, 1], 1)
        self.assertTrue(np.isfinite(z).all())

    def test_small_dataset_unknown_and_shape(self):
        p = Preprocessor().fit(np.array([[1.0], [2.0], [3.0]]), ["a"])
        self.assertTrue(np.isfinite(p.transform(np.array([[100.0], [np.inf]]))).all())
        with self.assertRaises(ValueError):
            p.transform(np.ones((2, 3)))

    def test_numeric_imputation_uses_training_median(self):
        p = Preprocessor("numeric").fit(np.arange(100).reshape(-1, 1), ["a"])
        z = p.transform(np.array([[np.nan], [49.5]]))
        np.testing.assert_array_equal(z[0], z[1])


class TestDataLoading(unittest.TestCase):
    def test_label_alignment_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "x_train.csv").write_text("Id,a,b\n10,1,\n11,2,3\n")
            (path / "y_train.csv").write_text("Id,_MICHD\n10,-1\n11,1\n")
            x, y, ids, names = load_training(path)
            self.assertEqual(names, ["a", "b"])
            np.testing.assert_array_equal(y, [0, 1])
            self.assertTrue(np.isnan(x[0, 1]))
            self.assertEqual(x.dtype, np.float32)
            x2, *_ = load_training(path)
            np.testing.assert_array_equal(x, x2)
            (path / "x_train.csv").write_text("Id,a,b\n10,8,\n11,2,3\n")
            self.assertEqual(load_training(path)[0][0, 0], 8)
            (path / "y_train.csv").write_text("Id,_MICHD\n11,-1\n10,1\n")
            with self.assertRaisesRegex(ValueError, "IDs"):
                load_training(path)

    def test_invalid_label_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "x_train.csv").write_text("Id,a\n1,2\n")
            (path / "y_train.csv").write_text("Id,_MICHD\n1,0\n")
            with self.assertRaisesRegex(ValueError, "labels"):
                load_training(path)
            (path / "x_test.csv").write_text("Id,a\n1,2\n1,3\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_test(path)


if __name__ == "__main__":
    unittest.main()
