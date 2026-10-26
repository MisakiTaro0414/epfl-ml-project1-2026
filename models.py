"""NumPy logistic optimization and classification evaluation utilities."""

import numpy as np


def sigmoid(z):
    """Stable sigmoid, including very large finite logits."""
    return np.exp(-np.logaddexp(0.0, -z))


def fit_logistic(x, y, lambda_=1e-4, max_iters=150, tol=1e-6):
    """Minimize mean BCE + lambda*||w[1:]||² using limited-memory BFGS.

    Column zero must be the intercept. Armijo backtracking checks every step.
    The optimizer is implemented here using only NumPy, with ten history pairs.
    """
    x = np.asarray(x, dtype=np.float32, order="C")
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(x.shape[1], dtype=np.float64)
    prevalence = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    w[0] = np.log(prevalence / (1 - prevalence))

    def objective(weights):
        logits = x @ weights.astype(np.float32)
        loss = np.mean(np.logaddexp(0.0, logits.astype(np.float64)) - y * logits)
        residual = (sigmoid(logits) - y).astype(np.float32)
        gradient = (x.T @ residual).astype(np.float64) / len(y)
        gradient[1:] += 2 * lambda_ * weights[1:]
        return float(loss + lambda_ * np.dot(weights[1:], weights[1:])), gradient

    value, gradient = objective(w)
    history = []
    trace = [value]
    status = "max_iters"
    for _ in range(max_iters):
        if np.max(np.abs(gradient)) <= tol:
            status = "gradient_tolerance"
            break
        direction = gradient.copy()
        coefficients = []
        for s, delta, rho in reversed(history):
            alpha = rho * np.dot(s, direction)
            coefficients.append(alpha)
            direction -= alpha * delta
        if history:
            s, delta, _ = history[-1]
            direction *= np.dot(s, delta) / np.dot(delta, delta)
        for (s, delta, rho), alpha in zip(history, reversed(coefficients)):
            direction += s * (alpha - rho * np.dot(delta, direction))
        direction *= -1
        slope = np.dot(gradient, direction)
        if slope >= 0:
            history = []
            direction = -gradient
            slope = -np.dot(gradient, gradient)
        step = 1.0
        for _ in range(25):
            candidate = w + step * direction
            new_value, new_gradient = objective(candidate)
            if np.isfinite(new_value) and new_value <= value + 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            status = "line_search_precision_limit"
            break
        s = candidate - w
        delta = new_gradient - gradient
        curvature = np.dot(s, delta)
        if curvature > 1e-12:
            history.append((s, delta, 1 / curvature))
            history = history[-10:]
        w, value, gradient = candidate, new_value, new_gradient
        trace.append(value)
    return w, {
        "iterations": len(trace) - 1,
        "status": status,
        "objective": value,
        "gradient_max": float(np.max(np.abs(gradient))),
        "loss_trace": trace,
    }


def best_f1_threshold(y, scores):
    """Find the best score >= threshold rule, treating tied scores together."""
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if not len(y) or y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Expected equal-length, nonempty labels and finite scores")
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    tp = np.cumsum(y[order])
    endpoints = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(y) - 1]
    f1 = 2 * tp[endpoints] / np.maximum(endpoints + 1 + y.sum(), 1)
    best = endpoints[int(np.argmax(f1))]
    return float(ranked[best])


def roc_auc(y, scores):
    """Mann--Whitney AUROC with average ranks for ties."""
    y, scores = np.asarray(y, dtype=int), np.asarray(scores)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranked = scores[order]
    starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return float(
        (ranks[y[order] == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def metrics(y, scores, threshold):
    """Binary metrics for positive label one, with explicit confusion counts."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": (tp + tn) / len(y),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "auc": roc_auc(y, scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
    }


def stratified_split(y, seed=433):
    """Disjoint 60/20/20 train/tuning/holdout split stratified by class."""
    rng = np.random.default_rng(seed)
    partitions = [[], [], []]
    for label in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == label))
        first, second = int(0.6 * len(idx)), int(0.8 * len(idx))
        for target, part in zip(partitions, np.split(idx, [first, second])):
            target.extend(part.tolist())
    return tuple(rng.permutation(np.asarray(part, dtype=int)) for part in partitions)
