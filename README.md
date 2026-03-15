momentum-backtest :

Backtest of a momentum strategy on S&P 500 and CAC 40. I built this as a personal project while studying quantitative finance alongside my degree wanted to see if I could take the stats and maths I was learning and turn them into something that actually runs on market data.
I kept the strategy simple on purpose. The goal was to understand each piece properly.

How it works

The signal is a dual moving average crossover. I go long when the 20-day MA crosses above the 60-day MA and stay flat otherwise. To avoid betting the same amount regardless of market conditions, I scale position sizes using volatility targeting: when markets are turbulent the position shrinks automatically, when they're calm it increases. I also apply a 5 bps transaction cost on every position change and shift the signal by one day to avoid lookahead bias.

Results:
Tested on 5 years of simulated data generated with GBM, with a crisis period injected around year 2.5 to stress-test the strategy.


```
                           S&P 500     CAC 40
------------------------------------------------
ann return                    8.7%       4.3%
ann vol                       8.2%       8.5%
sharpe                        1.06       0.51
max drawdown                -10.1%      -8.5%
bench sharpe                  0.63       0.83
```

The S&P 500 result is decent. Sharpe above 1, drawdown contained. CAC 40 is weaker, which makes sense since momentum tends to work better on trending markets. The win rate is ~31% on both, which is expected for a trend-following strategy: few big wins, many small losses.

![dashboard](performance_dashboard.png)

## Run it

```bash
pip install -r requirements.txt
python strategy.py
```

No external data API needed. Prices are generated internally for reproducibility.

## What I'd add next

- pull real index data via yfinance instead of simulated prices
- try RSI as a second filter to reduce false signals
- test different MA window pairs to see how sensitive the results are
- proper walk-forward validation instead of a single in-sample backtest

## Stack

Python, pandas, numpy, matplotlib.

Built by Haroun, bachelor business engineering, UCLouvain
