Momentum Backtest

A small personal project I started to learn more about quantitative finance and practice Python.
I wanted to see how a simple investment strategy based on moving averages would perform compared to just holding an investment.

The idea:
The strategy uses two moving averages: 20 days and 60 days.
When the 20-day average is higher than the 60-day average, the strategy invests. Otherwise, it stays out of the market.
I used the previous day’s signal to calculate returns and included a transaction cost of 0.05% when the position changes.

Data:
I used simulated prices for this project, not real market data.
The two simulations are labelled S&P 500 and CAC 40, but they don’t represent the actual indices.

Results:
The code compares the strategy with buy-and-hold using a few basic measures:

* Annualized return
* Volatility
* Sharpe ratio
* Maximum drawdown

The results are printed in the terminal.

Running the code:

Install the libraries:

pip install -r requirements.txt

Run the script:

python strategy.py

What I learned :
This project was mostly a way for me to practice Python and understand how a basic backtest works.
I also got to explore some performance measures that I had come across during my studies.
Since the prices are simulated, the results don’t really say anything about how the strategy would perform in real markets.

