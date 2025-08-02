import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Config ---
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2010-01-01'
end_date = '2025-01-01'
risk_free_rate = 0.02
penalty_cost = 0.00  # 20% cost if selling MSFT or TSLA below base

# --- Download data ---
data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# --- Indexes for stocks with penalty ---
penalized_stocks = {'MSFT': 0.50, 'TSLA': 0.40}  # default holdings
penalized_indices = {tickers.index(t): w for t, w in penalized_stocks.items()}

# --- Portfolio performance function ---
def portfolio_performance(weights):
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(weights.T @ cov_matrix.values @ weights)
    return ret, std

# --- Objective function: Negative Sharpe with penalty for selling penalized stocks ---
def neg_sharpe_with_penalty(weights, base_holdings, penalty):
    ret, std = portfolio_performance(weights)
    sharpe = (ret - risk_free_rate) / std
    penalty_term = 0
    for idx, base_w in base_holdings.items():
        if weights[idx] < base_w:
            penalty_term += penalty * (base_w - weights[idx])
    return -(sharpe - penalty_term)

# --- Constraints & bounds ---
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in tickers)
initial_guess = len(tickers) * [1. / len(tickers)]

# --- Sweep parameters for MSFT and TSLA base holdings ---
msft_range = np.linspace(0, 0.5, 51)
tsla_range = np.linspace(0, 0.3, 31)

# Storage for results
sharpe_matrix = np.zeros((len(msft_range), len(tsla_range)))
msft_weights_matrix = np.zeros_like(sharpe_matrix)
tsla_weights_matrix = np.zeros_like(sharpe_matrix)

for i, msft_base in enumerate(msft_range):
    for j, tsla_base in enumerate(tsla_range):
        base_holdings = {
            tickers.index('MSFT'): msft_base,
            tickers.index('TSLA'): tsla_base
        }
        # Check sum of base holdings < 1.0
        if msft_base + tsla_base >= 1.0:
            sharpe_matrix[i, j] = np.nan  # invalid
            continue

        # Run optimization
        result = minimize(
            neg_sharpe_with_penalty,
            initial_guess,
            args=(base_holdings, penalty_cost),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights_opt = result.x
            ret_opt, std_opt = portfolio_performance(weights_opt)
            sharpe_opt = (ret_opt - risk_free_rate) / std_opt

            sharpe_matrix[i, j] = sharpe_opt
            msft_weights_matrix[i, j] = weights_opt[tickers.index('MSFT')]
            tsla_weights_matrix[i, j] = weights_opt[tickers.index('TSLA')]
            print(f"MSFT base: {msft_base:.2f}, TSLA base: {tsla_base:.2f} => Sharpe Ratio: {sharpe_opt:.4f}")
        else:
            sharpe_matrix[i, j] = np.nan

# After completing the loops and filling sharpe_matrix

max_sharpe = np.nanmax(sharpe_matrix)
max_pos = np.unravel_index(np.nanargmax(sharpe_matrix), sharpe_matrix.shape)
optimal_msft_base = msft_range[max_pos[0]]
optimal_tsla_base = tsla_range[max_pos[1]]

print(f"\n✅ Optimal Sharpe Ratio: {max_sharpe:.4f}")
print(f"📌 Achieved at MSFT base holding = {optimal_msft_base:.2%}, TSLA base holding = {optimal_tsla_base:.2%}")

# --- Plot heatmap ---
plt.figure(figsize=(10, 7))
X, Y = np.meshgrid(tsla_range, msft_range)
c = plt.pcolormesh(X, Y, sharpe_matrix, shading='auto', cmap='viridis')
plt.colorbar(c, label='Sharpe Ratio')
plt.xlabel('TSLA Base Holding (fixed)')
plt.ylabel('MSFT Base Holding (fixed)')
plt.title('Sharpe Ratio vs Fixed MSFT and TSLA Holdings\n(20% Selling Penalty Applied)')
plt.show()