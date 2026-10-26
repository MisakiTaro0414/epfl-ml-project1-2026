"""Numerical and edge-case checks beyond the supplied course examples."""

import unittest

import numpy as np

import implementations as impl


class ImplementationsTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1.0, -2.0], [1.0, 0.0], [1.0, 3.0]])
        self.y = np.array([0.0, 0.0, 1.0])
        self.w = np.array([0.4, -0.2])

    def test_zero_iterations_preserve_weights_and_compute_loss(self):
        for function in (
            impl.mean_squared_error_gd,
            impl.mean_squared_error_sgd,
            impl.logistic_regression,
        ):
            w, loss = function(self.y, self.x, self.w, 0, 0.1)
            np.testing.assert_array_equal(w, self.w)
            self.assertFalse(np.shares_memory(w, self.w))
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(np.isfinite(loss))

    def test_initial_weights_are_not_mutated(self):
        original = self.w.copy()
        for function in (
            impl.mean_squared_error_gd,
            impl.mean_squared_error_sgd,
            impl.logistic_regression,
        ):
            function(self.y, self.x, self.w, 5, 0.01)
            np.testing.assert_array_equal(self.w, original)
        impl.reg_logistic_regression(self.y, self.x, 0.3, self.w, 5, 0.01)
        np.testing.assert_array_equal(self.w, original)

    def test_logistic_gradient_matches_finite_differences(self):
        penalty, step, epsilon = 0.17, 0.03, 1e-6

        def objective(w):
            z = self.x @ w
            return np.mean(np.logaddexp(0, z) - self.y * z) + penalty * (w @ w)

        numeric = np.empty_like(self.w)
        for j in range(self.w.size):
            direction = np.zeros_like(self.w)
            direction[j] = epsilon
            numeric[j] = (
                objective(self.w + direction) - objective(self.w - direction)
            ) / (2 * epsilon)
        w, loss = impl.reg_logistic_regression(self.y, self.x, penalty, self.w, 1, step)
        np.testing.assert_allclose((self.w - w) / step, numeric, atol=1e-9)
        self.assertAlmostEqual(loss, objective(w) - penalty * (w @ w), places=13)

    def test_mse_gradient_matches_finite_differences(self):
        step, epsilon = 0.03, 1e-6

        def objective(w):
            return np.mean((self.x @ w - self.y) ** 2) / 2

        numeric = np.empty_like(self.w)
        for j in range(self.w.size):
            direction = np.zeros_like(self.w)
            direction[j] = epsilon
            numeric[j] = (
                objective(self.w + direction) - objective(self.w - direction)
            ) / (2 * epsilon)
        w, loss = impl.mean_squared_error_gd(self.y, self.x, self.w, 1, step)
        np.testing.assert_allclose((self.w - w) / step, numeric, atol=1e-9)
        self.assertAlmostEqual(loss, objective(w), places=13)

    def test_logistic_extreme_logits_are_finite(self):
        x = np.array([[1000.0], [-1000.0], [1000.0], [-1000.0]])
        y = np.array([1.0, 0.0, 0.0, 1.0])
        with np.errstate(over="raise", invalid="raise"):
            w, loss = impl.logistic_regression(y, x, np.ones(1), 0, 0.001)
            self.assertEqual(loss, 500.0)
            w, loss = impl.logistic_regression(y, x, w, 1, 0.001)
            self.assertTrue(np.isfinite(w).all())
            self.assertTrue(np.isfinite(loss))

    def test_singular_normal_equations(self):
        x = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        y = np.array([3.0, 6.0, 9.0])
        for function in (
            impl.least_squares,
            lambda y, x: impl.ridge_regression(y, x, 0),
        ):
            w, loss = function(y, x)
            np.testing.assert_allclose(x @ w, y, atol=1e-12)
            np.testing.assert_allclose(w, [0.6, 1.2], atol=1e-12)
            self.assertLess(loss, 1e-24)

    def test_ridge_optimality_and_unpenalized_loss(self):
        penalty = 0.3
        w, loss = impl.ridge_regression(self.y, self.x, penalty)
        gradient = self.x.T @ (self.x @ w - self.y) / self.y.size + 2 * penalty * w
        np.testing.assert_allclose(gradient, 0, atol=1e-14)
        self.assertAlmostEqual(loss, np.mean((self.x @ w - self.y) ** 2) / 2)

    def test_duplicate_rows_do_not_rescale_objective(self):
        doubled_x, doubled_y = np.tile(self.x, (2, 1)), np.tile(self.y, 2)
        for function in (
            lambda y, x: impl.mean_squared_error_gd(y, x, self.w, 4, 0.1),
            lambda y, x: impl.reg_logistic_regression(y, x, 0.2, self.w, 4, 0.1),
            lambda y, x: impl.ridge_regression(y, x, 0.2),
        ):
            w, loss = function(self.y, self.x)
            doubled_w, doubled_loss = function(doubled_y, doubled_x)
            np.testing.assert_allclose(w, doubled_w, atol=1e-14)
            self.assertAlmostEqual(loss, doubled_loss, places=14)

    def test_sgd_single_row_update_and_full_data_loss(self):
        seed, step = 123, 0.05
        np.random.seed(seed)
        index = np.random.randint(self.y.size)
        expected = self.w - step * self.x[index] * (
            self.x[index] @ self.w - self.y[index]
        )
        np.random.seed(seed)
        w, loss = impl.mean_squared_error_sgd(self.y, self.x, self.w, 1, step)
        np.testing.assert_allclose(w, expected)
        self.assertAlmostEqual(loss, np.mean((self.x @ expected - self.y) ** 2) / 2)

    def test_sgd_seed_reproduces_results(self):
        np.random.seed(7)
        first = impl.mean_squared_error_sgd(self.y, self.x, self.w, 50, 0.01)
        np.random.seed(7)
        second = impl.mean_squared_error_sgd(self.y, self.x, self.w, 50, 0.01)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_invalid_targets_and_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            impl.logistic_regression(self.y * 2 - 1, self.x, self.w, 2, 0.1)
        with self.assertRaises(ValueError):
            impl.least_squares(self.y[:, None], self.x)
        with self.assertRaises(ValueError):
            impl.ridge_regression(self.y, self.x, -0.1)


if __name__ == "__main__":
    unittest.main()
