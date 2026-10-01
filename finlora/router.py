"""Confidence cascade: answer with the cheapest model unless its confidence is below a threshold."""
import numpy as np


def cascade(probs_small, probs_big, threshold):
    """Return (predictions, fraction escalated to the big model)."""
    esc = probs_small.max(1) < threshold
    pred = np.where(esc, probs_big.argmax(1), probs_small.argmax(1))
    return pred, float(esc.mean())


def cascade_curve(probs_small, probs_big, y, cost_small, cost_big, thresholds):
    rows = []
    for t in thresholds:
        pred, frac = cascade(probs_small, probs_big, t)
        rows.append(dict(threshold=float(t), escalated=frac, accuracy=float((pred == y).mean()),
                         avg_cost=cost_small + frac * cost_big))
    return rows
