import numpy as np
import pandas as pd
from src.factor_model import FactorRiskModel
from src.optimizer import ConvexPortfolioOptimizer
from data.pull_data import get_asset_returns, get_fama_french_factors

def run_pipeline():
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "BRK-B"]
    start_date = "2021-01-01"
    end_date = "2026-09-01"

    print("Fetching asset returns and Fama-French factors...")
    asset_ret = get_asset_returns(universe, start_date, end_date)
    ff = get_fama_french_factors(start_date, end_date)
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    aligned = asset_ret.join(ff[factor_cols], how="inner").dropna()
    aligned.to_csv("aligned_factor_returns.csv")
    print(f"Data cached: {aligned.shape[0]} trading days.")

    dates = aligned.index
    rebalance_dates = dates[::21]

    n_assets = len(universe)
    weights_history = []
    current_weights = np.ones(n_assets) / n_assets

    optimizer = ConvexPortfolioOptimizer(risk_aversion=2.5, transaction_cost_rate=0.0005)

    print("Executing walk-forward rebalancing...")
    for t in range(12, len(rebalance_dates)):
        lookback_idx = (dates >= rebalance_dates[t - 12]) & (dates < rebalance_dates[t])
        period_data = aligned.loc[lookback_idx]

        r_assets = period_data[universe]
        r_factors = period_data[factor_cols]

        model = FactorRiskModel(factor_names=factor_cols).fit(r_assets, r_factors)
        mu = r_assets.ewm(span=63).mean().iloc[-1].values * 252

        new_weights = optimizer.optimize_weights(
            expected_returns=mu,
            cov_matrix=model.total_cov * 252,
            prev_weights=current_weights,
            max_turnover=0.30,
            max_weight=0.25
        )
        current_weights = new_weights
        weights_history.append({"Date": rebalance_dates[t], **dict(zip(universe, current_weights))})

    weights_df = pd.DataFrame(weights_history).set_index("Date")
    print("\n--- Latest Optimal Portfolio Allocation ---")
    print(weights_df.iloc[-1].round(4))
    weights_df.to_csv("optimal_weights_history.csv")
    print("\nSaved allocation history to 'optimal_weights_history.csv'.")

if __name__ == "__main__":
    run_pipeline()