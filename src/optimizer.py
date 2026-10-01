import cvxpy as cp
import numpy as np

class ConvexPortfolioOptimizer:
    def __init__(self, risk_aversion: float = 1.0, transaction_cost_rate: float = 0.0010):
        self.risk_aversion = risk_aversion
        self.tc_rate = transaction_cost_rate

    def optimize_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        prev_weights: np.ndarray = None,
        max_turnover: float = None,
        max_weight: float = 0.25
    ) -> np.ndarray:
        n = len(expected_returns)
        if prev_weights is None:
            prev_weights = np.ones(n) / n

        w = cp.Variable(n)

        risk = cp.quad_form(w, cp.psd_wrap(cov_matrix))
        returns = expected_returns @ w
        turnover = cp.norm1(w - prev_weights)

        objective = cp.Minimize(0.5 * risk - self.risk_aversion * returns + self.tc_rate * turnover)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= max_weight
        ]

        if max_turnover is not None:
            constraints.append(turnover <= max_turnover)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Convex optimization failed with status: {prob.status}")

        raw_w = np.clip(w.value, 0.0, max_weight)
        return raw_w / np.sum(raw_w)