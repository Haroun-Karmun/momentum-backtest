import numpy as np
import pandas as pd


def generate_prices(seed, n_days=1260):
    np.random.seed(seed)

    returns = np.random.normal(0.0003, 0.012, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")

    return pd.Series(prices, index=dates)


def moving_average_strategy(prices):
    ma20 = prices.rolling(20).mean()
    ma60 = prices.rolling(60).mean()

    signal = (ma20 > ma60).astype(int)
    signal[ma60.isna()] = 0

    return signal


def backtest(prices):
    market_returns = prices.pct_change().fillna(0)
    signal = moving_average_strategy(prices)

    # Use the previous day's signal
    position = signal.shift(1).fillna(0)

    strategy_returns = position * market_returns

    cost = 0.0005
    trades = position.diff().abs().fillna(0)
    strategy_returns = strategy_returns - trades * cost

    return strategy_returns, market_returns


def performance_metrics(returns):
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1

    volatility = returns.std() * np.sqrt(252)

    if volatility > 0:
        sharpe = returns.mean() * 252 / volatility
    else:
        sharpe = 0

    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return annual_return, volatility, sharpe, max_drawdown


markets = {
    "S&P 500 (simulated)": 42,
    "CAC 40 (simulated)": 123
}

for name, seed in markets.items():
    prices = generate_prices(seed)

    strategy_returns, market_returns = backtest(prices)

    strategy = performance_metrics(strategy_returns)
    market = performance_metrics(market_returns)

    print(f"\n{name}")
    print("-" * 40)

    print(f"Annual return:     {strategy[0]:.2%} vs {market[0]:.2%}")
    print(f"Volatility:        {strategy[1]:.2%} vs {market[1]:.2%}")
    print(f"Sharpe ratio:      {strategy[2]:.2f} vs {market[2]:.2f}")
    print(f"Maximum drawdown:  {strategy[3]:.2%} vs {market[3]:.2%}")
