"""The six required learning algorithms, implemented using only NumPy.

Regression losses are mean squared errors divided by two. Logistic losses are
mean binary cross-entropies. Regularized objectives add ``lambda_ * (w @ w)``,
including any intercept supplied by the caller. Returned losses always omit
that penalty and are evaluated at the returned weights.
"""

import operator

import numpy as np


def _data(y, tx):
    """Validate the assignment's (N,) targets and (N, D) design matrix."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    if y.ndim != 1 or tx.ndim != 2 or tx.shape[0] != y.size:
        raise ValueError("Expected y with shape (N,) and tx with shape (N, D).")
    if y.size == 0 or tx.shape[1] == 0:
        raise ValueError("The design matrix must have rows and columns.")
    if not np.isfinite(y).all() or not np.isfinite(tx).all():
        raise ValueError("Inputs must be finite; preprocess missing values first.")
    return y, tx


def _initial_weights(tx, initial_w, max_iters, gamma):
    """Copy weights so iterative methods never mutate their caller's array."""
    w = np.array(initial_w, dtype=float, copy=True)
    if w.ndim != 1 or w.size != tx.shape[1] or not np.isfinite(w).all():
        raise ValueError("initial_w must be a finite vector of shape (D,).")
    max_iters = operator.index(max_iters)
    if max_iters < 0 or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("max_iters and gamma must be nonnegative and finite.")
    return w, max_iters


def _mse(y, tx, w):
    residual = tx @ w - y
    return np.mean(residual * residual) / 2.0


def _sigmoid(z):
    """Evaluate the sigmoid without overflow for large positive/negative z."""
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    nonnegative = z >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
    exp_z = np.exp(z[~nonnegative])
    result[~nonnegative] = exp_z / (1.0 + exp_z)
    return result


def _logistic_loss(y, tx, w):
    # Binary y makes this equivalent to logaddexp(0, z) - y*z, while
    # avoiding cancellation when a confident positive prediction is correct.
    return np.mean(np.logaddexp(0.0, (1.0 - 2.0 * y) * (tx @ w)))


def _validate_lambda(lambda_):
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be nonnegative and finite.")


def _solve_normal_equation(gram, rhs):
    """Solve normal equations, with a minimum-norm fallback for singular data."""
    if np.linalg.matrix_rank(gram) < gram.shape[0]:
        return np.linalg.pinv(gram, hermitian=True) @ rhs
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram, hermitian=True) @ rhs


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Minimize mean squared error / 2 with full-batch gradient descent.

    Run exactly ``max_iters`` updates using gradient ``tx.T @ (tx @ w-y)/N``.
    Return the final weights and the full-data loss at those weights.
    """
    y, tx = _data(y, tx)
    w, max_iters = _initial_weights(tx, initial_w, max_iters, gamma)
    for _ in range(max_iters):
        w -= gamma * (tx.T @ (tx @ w - y)) / y.size
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Minimize mean squared error / 2 with batch-size-one SGD.

    Each update samples one row uniformly with replacement. Call
    ``np.random.seed(seed)`` before this function to reproduce sampling.
    The returned loss is evaluated on all rows, not only the last sampled row.
    """
    y, tx = _data(y, tx)
    w, max_iters = _initial_weights(tx, initial_w, max_iters, gamma)
    for _ in range(max_iters):
        index = np.random.randint(y.size)
        row = tx[index]
        w -= gamma * row * (row @ w - y[index])
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Solve the least-squares normal equations and return (w, MSE / 2).

    Singular normal equations use their pseudoinverse. No call to
    ``numpy.linalg.lstsq`` is made.
    """
    y, tx = _data(y, tx)
    w = _solve_normal_equation(tx.T @ tx, tx.T @ y)
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Minimize MSE / 2 + lambda_ * ||w||^2 by normal equations.

    Solve ``(tx.T @ tx + 2*N*lambda_*I) @ w = tx.T @ y``. The returned
    loss excludes the regularization penalty, as required by the assignment.
    """
    y, tx = _data(y, tx)
    _validate_lambda(lambda_)
    gram = tx.T @ tx + 2.0 * y.size * lambda_ * np.eye(tx.shape[1])
    w = _solve_normal_equation(gram, tx.T @ y)
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit binary logistic regression with full-batch gradient descent.

    Targets must be 0 or 1. The objective is mean binary cross-entropy.
    Return the final weights and the cross-entropy evaluated at those weights.
    """
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit logistic regression with the penalty lambda_ * ||w||^2.

    Gradient updates use mean cross-entropy plus ``2*lambda_*w``. Targets
    must be 0 or 1; the returned mean cross-entropy excludes the penalty.
    """
    y, tx = _data(y, tx)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Logistic regression requires labels in {0, 1}.")
    _validate_lambda(lambda_)
    w, max_iters = _initial_weights(tx, initial_w, max_iters, gamma)
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / y.size
        w -= gamma * (gradient + 2.0 * lambda_ * w)
    return w, _logistic_loss(y, tx, w)
