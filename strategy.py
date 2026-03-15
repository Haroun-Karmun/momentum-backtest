"""
Momentum Trading Strategy — S&P 500 / CAC 40
=============================================
Moving average crossover with volatility-adjusted position sizing.
Backtest engine with performance metrics and visualisation.

Author : Haroun
Project : Personal research — quantitative finance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta


# ── 1. SIMULATED PRICE DATA ───────────────────────────────────────────────────

def generate_index_prices(name, seed, n_days=1260, mu=0.00035, sigma=0.012):
    """
    Simulate daily closing prices using Geometric Brownian Motion (GBM).
    GBM is the standard model for equity price dynamics in finance.

    Parameters
    ----------
    name    : index name (for labelling)
    seed    : random seed for reproducibility
    n_days  : number of trading days (~5 years)
    mu      : daily drift (annualised ~8.8%)
    sigma   : daily volatility (annualised ~19%)
    """
    np.random.seed(seed)
    returns = np.random.normal(mu, sigma, n_days)

    # Add a mild crisis period to make the data realistic
    crisis_start = int(n_days * 0.45)
    returns[crisis_start : crisis_start + 60] -= 0.008

    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range(start="2019-01-02", periods=n_days, freq="B")
    return pd.Series(prices, index=dates, name=name)


# ── 2. SIGNALS ────────────────────────────────────────────────────────────────

def compute_signals(prices, short_window=20, long_window=60):
    """
    Dual moving average crossover signal.

    Logic:
      - BUY  (signal = 1) when short MA crosses above long MA → upward momentum
      - SELL (signal = 0) when short MA crosses below long MA → downward momentum

    This is one of the oldest and most studied momentum indicators.
    """
    ma_short = prices.rolling(short_window).mean()
    ma_long  = prices.rolling(long_window).mean()

    signal = (ma_short > ma_long).astype(int)
    signal = signal.shift(1)  # shift by 1 day to avoid lookahead bias

    return signal, ma_short, ma_long


# ── 3. POSITION SIZING ────────────────────────────────────────────────────────

def volatility_target(returns, signal, target_vol=0.10, window=20):
    """
    Scale position size so the strategy targets a constant annual volatility.

    Instead of always going 100% in or out, we invest a fraction proportional
    to (target_vol / realised_vol). This reduces exposure when markets are
    turbulent — a simple form of risk management.
    """
    realised_vol = returns.rolling(window).std() * np.sqrt(252)
    realised_vol = realised_vol.replace(0, np.nan).ffill()

    weight = (target_vol / realised_vol).clip(0, 1.5)
    sized_signal = signal * weight
    return sized_signal


# ── 4. BACKTEST ───────────────────────────────────────────────────────────────

def backtest(prices, short_window=20, long_window=60, cost_bps=5):
    """
    Run the full backtest and return a results dictionary.

    cost_bps : transaction cost in basis points (1 bps = 0.01%)
               applied each time the position changes
    """
    daily_returns = prices.pct_change()

    signal, ma_short, ma_long = compute_signals(prices, short_window, long_window)
    sized = volatility_target(daily_returns, signal)

    # Strategy return = sized position × next day's market return
    strat_returns = sized.shift(1) * daily_returns

    # Transaction costs: apply on days where position changes
    position_change = sized.diff().abs()
    costs = position_change * (cost_bps / 10_000)
    strat_returns = strat_returns - costs

    # Cumulative performance
    equity_curve  = (1 + strat_returns).cumprod()
    market_curve  = (1 + daily_returns).cumprod()

    return {
        "prices"       : prices,
        "daily_ret"    : daily_returns,
        "strat_ret"    : strat_returns,
        "equity_curve" : equity_curve,
        "market_curve" : market_curve,
        "signal"       : signal,
        "sized"        : sized,
        "ma_short"     : ma_short,
        "ma_long"      : ma_long,
    }


# ── 5. PERFORMANCE METRICS ────────────────────────────────────────────────────

def compute_metrics(strat_ret, market_ret, label="Strategy"):
    """
    Compute standard quantitative finance performance metrics.
    """
    ann = 252  # trading days per year

    # Annualised return
    total_return   = (1 + strat_ret).prod() - 1
    n_years        = len(strat_ret) / ann
    ann_return     = (1 + total_return) ** (1 / n_years) - 1

    # Volatility
    ann_vol        = strat_ret.std() * np.sqrt(ann)

    # Sharpe Ratio (risk-free rate assumed 0 for simplicity)
    sharpe         = ann_return / ann_vol if ann_vol > 0 else 0

    # Maximum Drawdown
    equity         = (1 + strat_ret).cumprod()
    rolling_max    = equity.cummax()
    drawdown       = (equity - rolling_max) / rolling_max
    max_drawdown   = drawdown.min()

    # Calmar Ratio = annualised return / abs(max drawdown)
    calmar         = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate       = (strat_ret > 0).mean()

    # Benchmark comparison
    mkt_ann_ret    = (1 + market_ret).prod() ** (1 / n_years) - 1
    mkt_ann_vol    = market_ret.std() * np.sqrt(ann)
    mkt_sharpe     = mkt_ann_ret / mkt_ann_vol if mkt_ann_vol > 0 else 0

    return {
        "label"        : label,
        "ann_return"   : ann_return,
        "ann_vol"      : ann_vol,
        "sharpe"       : sharpe,
        "max_drawdown" : max_drawdown,
        "calmar"       : calmar,
        "win_rate"     : win_rate,
        "mkt_ann_ret"  : mkt_ann_ret,
        "mkt_sharpe"   : mkt_sharpe,
    }


# ── 6. VISUALISATION ──────────────────────────────────────────────────────────

def plot_results(results_list, metrics_list):
    """
    Generate a clean 4-panel performance dashboard.
    """
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(
        "Momentum Strategy — Backtesting Dashboard\nS&P 500 & CAC 40  |  MA Crossover + Volatility Targeting",
        fontsize=13, fontweight="bold", y=0.98, color="#1F3864"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    colors = {"S&P 500": "#1F3864", "CAC 40": "#C0392B"}

    # ── Panel 1 : Equity curves ──
    ax1 = fig.add_subplot(gs[0, 0])
    for res, m in zip(results_list, metrics_list):
        label = m["label"]
        c = colors[label]
        ax1.plot(res["equity_curve"], color=c, linewidth=1.6, label=f"{label} strategy")
        ax1.plot(res["market_curve"], color=c, linewidth=0.8, linestyle="--", alpha=0.5, label=f"{label} buy & hold")
    ax1.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_title("Cumulative Performance", fontsize=10, fontweight="bold", color="#1F3864")
    ax1.set_ylabel("Portfolio value (base 1)")
    ax1.legend(fontsize=7.5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2 : MA crossover signal on prices ──
    ax2 = fig.add_subplot(gs[0, 1])
    res0 = results_list[0]
    ax2.plot(res0["prices"], color="#1F3864", linewidth=1.0, label="Price", alpha=0.8)
    ax2.plot(res0["ma_short"], color="#E67E22", linewidth=1.2, label="MA 20")
    ax2.plot(res0["ma_long"],  color="#27AE60", linewidth=1.2, label="MA 60")
    # Shade long periods
    in_position = res0["signal"].fillna(0).astype(bool)
    ax2.fill_between(res0["prices"].index, res0["prices"].min(), res0["prices"].max(),
                     where=in_position, alpha=0.07, color="#1F3864")
    ax2.set_title("S&P 500 — MA Crossover Signal", fontsize=10, fontweight="bold", color="#1F3864")
    ax2.set_ylabel("Simulated price")
    ax2.legend(fontsize=7.5)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Panel 3 : Rolling Sharpe ratio (252-day) ──
    ax3 = fig.add_subplot(gs[1, 0])
    for res, m in zip(results_list, metrics_list):
        roll_sharpe = (
            res["strat_ret"].rolling(252).mean() /
            res["strat_ret"].rolling(252).std()
        ) * np.sqrt(252)
        ax3.plot(roll_sharpe, color=colors[m["label"]], linewidth=1.2, label=m["label"])
    ax3.axhline(0, color="gray", linewidth=0.6)
    ax3.axhline(1, color="green", linewidth=0.5, linestyle="--", alpha=0.5, label="Sharpe = 1")
    ax3.set_title("Rolling Sharpe Ratio (1-year window)", fontsize=10, fontweight="bold", color="#1F3864")
    ax3.set_ylabel("Sharpe ratio")
    ax3.legend(fontsize=7.5)
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Panel 4 : Drawdown ──
    ax4 = fig.add_subplot(gs[1, 1])
    for res, m in zip(results_list, metrics_list):
        equity    = res["equity_curve"]
        drawdown  = (equity - equity.cummax()) / equity.cummax()
        ax4.fill_between(drawdown.index, drawdown, 0,
                         alpha=0.35, color=colors[m["label"]], label=m["label"])
        ax4.plot(drawdown, color=colors[m["label"]], linewidth=0.8)
    ax4.set_title("Drawdown", fontsize=10, fontweight="bold", color="#1F3864")
    ax4.set_ylabel("Drawdown (%)")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax4.legend(fontsize=7.5)
    ax4.grid(axis="y", alpha=0.3)
    ax4.spines[["top", "right"]].set_visible(False)

    plt.savefig("performance_dashboard.png", dpi=150, bbox_inches="tight")
    print("Chart saved: performance_dashboard.png")
    plt.close()


# ── 7. METRICS TABLE ──────────────────────────────────────────────────────────

def print_metrics_table(metrics_list):
    print("\n" + "═" * 58)
    print(f"{'PERFORMANCE SUMMARY':^58}")
    print("═" * 58)
    header = f"{'Metric':<26} {'S&P 500':>14} {'CAC 40':>14}"
    print(header)
    print("─" * 58)

    rows = [
        ("Ann. Return (strategy)",  "ann_return",   "{:.1%}"),
        ("Ann. Volatility",         "ann_vol",       "{:.1%}"),
        ("Sharpe Ratio",            "sharpe",        "{:.2f}"),
        ("Max Drawdown",            "max_drawdown",  "{:.1%}"),
        ("Calmar Ratio",            "calmar",        "{:.2f}"),
        ("Win Rate",                "win_rate",      "{:.1%}"),
        ("Benchmark Ann. Return",   "mkt_ann_ret",   "{:.1%}"),
        ("Benchmark Sharpe",        "mkt_sharpe",    "{:.2f}"),
    ]

    for label, key, fmt in rows:
        vals = [fmt.format(m[key]) for m in metrics_list]
        print(f"{label:<26} {vals[0]:>14} {vals[1]:>14}")

    print("═" * 58)


# ── 8. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running momentum backtest on S&P 500 and CAC 40 simulated data...")
    print("Strategy : Dual MA crossover (20/60) + volatility targeting\n")

    # Generate simulated data
    sp500 = generate_index_prices("S&P 500", seed=42,  mu=0.00038, sigma=0.011)
    cac40 = generate_index_prices("CAC 40",  seed=99,  mu=0.00028, sigma=0.013)

    # Run backtests
    res_sp  = backtest(sp500)
    res_cac = backtest(cac40)

    # Compute metrics
    m_sp  = compute_metrics(res_sp["strat_ret"],  res_sp["daily_ret"],  label="S&P 500")
    m_cac = compute_metrics(res_cac["strat_ret"], res_cac["daily_ret"], label="CAC 40")

    # Print results
    print_metrics_table([m_sp, m_cac])

    # Plot
    plot_results([res_sp, res_cac], [m_sp, m_cac])

    print("\nDone.")
