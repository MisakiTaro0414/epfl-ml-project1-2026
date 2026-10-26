"""Independent checks of NumPy optimization, thresholding, and validation."""

import unittest

import numpy as np

from models import best_f1_threshold, fit_logistic, metrics, roc_auc, sigmoid
from models import stratified_split


def binary_objective(x, y, w, penalty):
    """Float64 reference objective; the intercept is not regularized."""
    logits = x @ w
    return np.mean(np.logaddexp(0, (1 - 2 * y) * logits)) + penalty * (w[1:] @ w[1:])


def newton_reference(x, y, penalty):
    """Small independent Newton solver used only as a testing oracle."""
    w = np.zeros(x.shape[1])
    regularizer = np.eye(x.shape[1]) * (2 * penalty)
    regularizer[0, 0] = 0
    for _ in range(40):
        probability = 1 / (1 + np.exp(-(x @ w)))
        gradient = x.T @ (probability - y) / y.size + regularizer @ w
        if np.max(np.abs(gradient)) < 1e-11:
            break
        curvature = probability * (1 - probability)
        hessian = (x.T * curvature) @ x / y.size + regularizer
        direction = np.linalg.solve(hessian, gradient)
        step = 1.0
        while binary_objective(x, y, w - step * direction, penalty) > binary_objective(
            x, y, w, penalty
        ):
            step *= 0.5
            if step < 1e-10:
                break
        w -= step * direction
    return w


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.x = np.column_stack([np.ones(500), rng.normal(size=(500, 4))])
        probability = 1 / (
            1 + np.exp(-(self.x @ np.array([-1.1, 1.2, -0.8, 0.4, 0.7])))
        )
        self.y = (rng.random(500) < probability).astype(int)

    def test_sigmoid_extreme_logits(self):
        with np.errstate(over="raise", invalid="raise"):
            probability = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_array_equal(probability, [0.0, 0.5, 1.0])

    def test_lbfgs_matches_independent_newton_solution(self):
        for penalty in (0.1, 0.001):
            w, info = fit_logistic(self.x, self.y, lambda_=penalty, max_iters=100)
            reference = newton_reference(self.x, self.y, penalty)
            np.testing.assert_allclose(w, reference, atol=8e-5, rtol=0)
            self.assertAlmostEqual(
                info["objective"],
                binary_objective(self.x, self.y, w, penalty),
                places=7,
            )
            self.assertTrue(np.all(np.diff(info["loss_trace"]) <= 1e-12))
            self.assertEqual(info["iterations"], len(info["loss_trace"]) - 1)

    def test_solution_is_stationary_by_finite_differences(self):
        penalty = 0.01
        w, info = fit_logistic(self.x, self.y, lambda_=penalty, max_iters=100)
        epsilon = 1e-5
        finite_difference = np.empty_like(w)
        for j in range(w.size):
            direction = np.zeros_like(w)
            direction[j] = epsilon
            finite_difference[j] = (
                binary_objective(self.x, self.y, w + direction, penalty)
                - binary_objective(self.x, self.y, w - direction, penalty)
            ) / (2 * epsilon)
        self.assertLess(np.max(np.abs(finite_difference)), 2e-5)
        self.assertAlmostEqual(
            np.max(np.abs(finite_difference)), info["gradient_max"], delta=2e-7
        )

    def test_intercept_is_not_regularized(self):
        y = np.r_[np.zeros(80), np.ones(20)]
        x = np.ones((100, 1))
        w, info = fit_logistic(x, y, lambda_=100)
        self.assertAlmostEqual(w[0], np.log(0.2 / 0.8), places=6)
        self.assertAlmostEqual(
            info["objective"], -0.2 * np.log(0.2) - 0.8 * np.log(0.8), places=7
        )

    def test_zero_iterations_and_determinism(self):
        original_x, original_y = self.x.copy(), self.y.copy()
        w, info = fit_logistic(self.x, self.y, max_iters=0)
        np.testing.assert_array_equal(w[1:], np.zeros(self.x.shape[1] - 1))
        self.assertEqual(info["iterations"], 0)
        w1, info1 = fit_logistic(self.x, self.y, max_iters=20)
        w2, info2 = fit_logistic(self.x, self.y, max_iters=20)
        np.testing.assert_array_equal(w1, w2)
        self.assertEqual(info1, info2)
        np.testing.assert_array_equal(self.x, original_x)
        np.testing.assert_array_equal(self.y, original_y)


class EvaluationTests(unittest.TestCase):
    def test_threshold_ties_are_indivisible(self):
        y = np.array([1, 0, 1, 0])
        scores = np.array([0.8, 0.8, 0.3, 0.3])
        threshold = best_f1_threshold(y, scores)
        self.assertEqual(threshold, 0.3)
        self.assertAlmostEqual(metrics(y, scores, threshold)["f1"], 2 / 3)

    def test_threshold_matches_exhaustive_search_with_ties(self):
        rng = np.random.default_rng(30)
        for _ in range(40):
            y = rng.integers(0, 2, size=51)
            scores = rng.integers(0, 8, size=y.size) / 7
            threshold = best_f1_threshold(y, scores)
            expected = max(metrics(y, scores, t)["f1"] for t in np.unique(scores))
            self.assertAlmostEqual(metrics(y, scores, threshold)["f1"], expected)

    def test_all_tied_scores_and_single_class(self):
        scores = np.full(5, 0.4)
        y = np.array([0, 0, 0, 0, 1])
        self.assertEqual(best_f1_threshold(y, scores), 0.4)
        self.assertEqual(roc_auc(y, scores), 0.5)
        self.assertTrue(np.isnan(roc_auc(np.zeros(5), scores)))
        self.assertTrue(np.isnan(roc_auc(np.ones(5), scores)))

    def test_auc_matches_pairwise_definition_including_ties(self):
        y = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        scores = np.array([0.1, 0.2, 0.2, 0.4, 0.6, 0.6, 0.3, 0.9, 0.4, 0.5])
        pairwise = scores[y == 1, None] - scores[y == 0][None, :]
        expected = np.mean((pairwise > 0) + 0.5 * (pairwise == 0))
        self.assertAlmostEqual(roc_auc(y, scores), expected)
        self.assertEqual(roc_auc(y, y), 1.0)
        self.assertEqual(roc_auc(y, 1 - y), 0.0)

    def test_imbalance_and_confusion_counts(self):
        y = np.r_[np.ones(10), np.zeros(990)]
        scores = np.r_[np.full(6, 0.8), np.full(4, 0.1), np.full(9, 0.8), np.zeros(981)]
        result = metrics(y, scores, 0.8)
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [6, 9, 4, 981])
        self.assertAlmostEqual(result["f1"], 12 / 25)
        self.assertAlmostEqual(result["accuracy"], 987 / 1000)
        self.assertAlmostEqual(result["precision"], 6 / 15)
        self.assertAlmostEqual(result["recall"], 6 / 10)
        no_positives = metrics(y, scores, 1.0)
        self.assertEqual(no_positives["precision"], 0)
        self.assertEqual(no_positives["f1"], 0)


class SplitTests(unittest.TestCase):
    def test_stratified_partition_is_disjoint_complete_and_reproducible(self):
        y = np.r_[np.zeros(97, dtype=int), np.ones(23, dtype=int)]
        partitions = stratified_split(y, seed=41)
        repeated = stratified_split(y, seed=41)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(partitions)), np.arange(y.size)
        )
        self.assertEqual(len(set(np.concatenate(partitions).tolist())), y.size)
        for first, second in zip(partitions, repeated):
            np.testing.assert_array_equal(first, second)
        for label in (0, 1):
            count = np.sum(y == label)
            first, second = int(0.6 * count), int(0.8 * count)
            self.assertEqual(
                [int(np.sum(y[part] == label)) for part in partitions],
                [first, second - first, count - second],
            )


if __name__ == "__main__":
    unittest.main()
