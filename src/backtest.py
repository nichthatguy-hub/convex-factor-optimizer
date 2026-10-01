import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_portfolio(
    weights_path: str = "optimal_weights_history.csv",
    data_path: str = "aligned_factor_returns.csv",
    output_plot: str = "tearsheet.png"
):
    weights_df = pd.read_csv(weights_path, index_col="Date", parse_dates=True)
    aligned_df = pd.read_csv(data_path, index_col="Date", parse_dates=True)

    assets = weights_df.columns.tolist()
    asset_returns = aligned_df[assets]
    market_returns = aligned_df["Mkt-RF"] + aligned_df["RF"]

    # Align dates
    start_date = weights_df.index[0]
    sub_returns = asset_returns.loc[start_date:].copy()

    # Rebalance weights forward-filled to daily
    daily_weights = weights_df.reindex(sub_returns.index).ffill()

    # Compute daily strategy returns (lagged by 1 day to prevent lookahead bias)
    strat_daily = (daily_weights.shift(1) * sub_returns).sum(axis=1).dropna()
    bench_daily = sub_returns.mean(axis=1).loc[strat_daily.index]
    mkt_daily = market_returns.loc[strat_daily.index]

    # Performance metrics
    cum_strat = (1 + strat_daily).cumprod()
    cum_bench = (1 + bench_daily).cumprod()
    cum_mkt = (1 + mkt_daily).cumprod()

    ann_ret_strat = strat_daily.mean() * 252
    ann_vol_strat = strat_daily.std() * np.sqrt(252)
    sharpe_strat = ann_ret_strat / ann_vol_strat if ann_vol_strat != 0 else 0

    ann_ret_bench = bench_daily.mean() * 252
    ann_vol_bench = bench_daily.std() * np.sqrt(252)
    sharpe_bench = ann_ret_bench / ann_vol_bench if ann_vol_bench != 0 else 0

    # Drawdown
    hwm = cum_strat.cummax()
    drawdown = (cum_strat - hwm) / hwm
    max_dd = drawdown.min()

    # Average Annual Turnover
    rebalance_diffs = weights_df.diff().abs().sum(axis=1).dropna()
    avg_annual_turnover = rebalance_diffs.mean() * 12  # 12 rebalances per year

    print("=====================================================")
    print("      CONVEX OPTIMIZER PERFORMANCE TEARSHEET         ")
    print("=====================================================")
    print(f"Annualized Return (Strategy):   {ann_ret_strat * 100:.2f}%")
    print(f"Annualized Volatility:          {ann_vol_strat * 100:.2f}%")
    print(f"Sharpe Ratio (Rf=0):            {sharpe_strat:.2f}")
    print(f"Max Drawdown:                   {max_dd * 100:.2f}%")
    print(f"Average Annual Turnover:        {avg_annual_turnover * 100:.2f}%")
    print("-----------------------------------------------------")
    print(f"Annualized Return (Equal-Wt):   {ann_ret_bench * 100:.2f}%")
    print(f"Sharpe Ratio (Equal-Wt):        {sharpe_bench:.2f}")
    print("=====================================================")

    # Plot visual tearsheet
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)

    # 1. Cumulative returns
    axes[0].plot(cum_strat.index, cum_strat, label="Convex Factor Optimizer", color="#1f77b4", lw=2)
    axes[0].plot(cum_bench.index, cum_bench, label="Equal-Weight 1/N Benchmark", color="#7f7f7f", linestyle="--")
    axes[0].plot(cum_mkt.index, cum_mkt, label="Market (Mkt)", color="#bcbd22", linestyle=":", alpha=0.7)
    axes[0].set_title("Cumulative Out-of-Sample Performance", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Growth of $1.00")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 2. Drawdown
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.35)
    axes[1].plot(drawdown.index, drawdown, color="#d62728", lw=1)
    axes[1].set_title("Strategy Historical Drawdown", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)

    # 3. Allocation stacked area chart
    axes[2].stackplot(weights_df.index, weights_df.values.T, labels=weights_df.columns, alpha=0.85)
    axes[2].set_title("Dynamic Portfolio Weights Over Time", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Portfolio Weight")
    axes[2].set_ylim(0, 1.0)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Tearsheet chart generated: {output_plot}")

if __name__ == "__main__":
    evaluate_portfolio()