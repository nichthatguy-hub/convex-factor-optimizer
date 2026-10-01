import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

class FactorRiskModel:
    def __init__(self, factor_names: list[str]):
        self.factor_names = factor_names
        self.betas: np.ndarray = None
        self.factor_cov: np.ndarray = None
        self.specific_var: np.ndarray = None
        self.total_cov: np.ndarray = None

    def fit(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> "FactorRiskModel":
        aligned = asset_returns.join(factor_returns[self.factor_names], how="inner").dropna()
        R = aligned[asset_returns.columns].values
        F = aligned[self.factor_names].values
        
        N = R.shape[1]
        K = F.shape[1]
        
        F_with_const = np.column_stack([np.ones(len(F)), F])
        betas = np.zeros((N, K))
        residuals = np.zeros_like(R)
        
        for i in range(N):
            b_hat = np.linalg.lstsq(F_with_const, R[:, i], rcond=None)[0]
            betas[i, :] = b_hat[1:]
            residuals[:, i] = R[:, i] - F_with_const @ b_hat

        self.betas = betas
        
        lw_factor = LedoitWolf().fit(F)
        self.factor_cov = lw_factor.covariance_
        
        self.specific_var = np.var(residuals, axis=0, ddof=K + 1)
        self.total_cov = self.betas @ self.factor_cov @ self.betas.T + np.diag(self.specific_var)
        return self

    def get_factor_loadings(self, asset_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(self.betas, index=asset_names, columns=self.factor_names)