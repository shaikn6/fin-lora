import numpy as np
from finlora.router import cascade, cascade_curve

small = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3], [0.34, 0.33, 0.33]])
big = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]])


def test_threshold_zero_never_escalates_and_one_always_does():
    p0, f0 = cascade(small, big, 0.0)
    p1, f1 = cascade(small, big, 1.01)
    assert f0 == 0.0 and (p0 == small.argmax(1)).all()
    assert f1 == 1.0 and (p1 == big.argmax(1)).all()


def test_only_low_confidence_rows_escalate():
    pred, frac = cascade(small, big, 0.5)
    assert frac == 2 / 3 and list(pred) == [0, 2, 0]


def test_cost_interpolates_between_models():
    rows = cascade_curve(small, big, np.array([0, 2, 0]), 1.0, 10.0, [0.0, 1.01])
    assert rows[0]["avg_cost"] == 1.0 and rows[1]["avg_cost"] == 11.0
