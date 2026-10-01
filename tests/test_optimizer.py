import numpy as np
import pytest
from src.optimizer import ConvexPortfolioOptimizer

def test_optimization_constraints():
    np.random.seed(42)
    n = 5
    mu = np.array([0.08, 0.12, 0.05, 0.15, 0.10])
    A = np.random.randn(n, n)
    cov = A @ A.T + np.eye(n) * 0.01

    prev_w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    optimizer = ConvexPortfolioOptimizer(risk_aversion=1.0)
    w = optimizer.optimize_weights(mu, cov, prev_weights=prev_w, max_turnover=0.20, max_weight=0.35)

    assert np.isclose(np.sum(w), 1.0), "Budget constraint violated: sum(w) != 1.0"
    assert np.all(w >= -1e-6), "No-shorting constraint violated: w_i < 0"
    assert np.all(w <= 0.35 + 1e-6), "Concentration limit violated: w_i > 0.35"
    assert np.sum(np.abs(w - prev_w)) <= 0.20 + 1e-4, "Turnover limit exceeded"