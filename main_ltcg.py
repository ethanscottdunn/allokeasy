import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --- Setup ---
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'VTSAX']
start_date = '2020-01-01'
end_date = '2025-01-01'
risk_free_rate = 0.02
ltcg_tax_rate = 0.20  # 20% LTCG tax on gains when selling

# --- Download price data ---
data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# --- Current holdings (your existing portfolio) ---
current_weights = np.array([0.00, 0.30, 0.10, 0.40, 0.20])  # example current weights summing to 1

def portfolio_performance(weights):
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(weights.T @ cov_matrix.values @ weights)
    return ret, std

def total_ltcg_tax_cost(current_w, new_w, tax_rate):
    # Calculate amount sold (only reductions)
    sold = np.clip(current_w - new_w, a_min=0, a_max=None)
    # Tax cost is 20% of amount sold (approximation)
    cost = np.sum(sold) * tax_rate
    return cost

def neg_sharpe_without_cost(weights):
    ret, std = portfolio_performance(weights)
    sharpe = (ret - risk_free_rate) / std
    return -sharpe

def neg_sharpe_with_ltcg_cost(weights):
    ret, std = portfolio_performance(weights)
    tax_cost = total_ltcg_tax_cost(current_weights, weights, ltcg_tax_rate)
    adjusted_ret = ret - tax_cost
    sharpe = (adjusted_ret - risk_free_rate) / std
    return -sharpe

# Constraints & bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in tickers)
initial_guess = current_weights.copy()

# Optimize
result = minimize(
    neg_sharpe_with_ltcg_cost,
    initial_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

if result.success:
    weights_opt = result.x
    ret_opt, std_opt = portfolio_performance(weights_opt)
    tax_cost_opt = total_ltcg_tax_cost(current_weights, weights_opt, ltcg_tax_rate)
    adjusted_ret_opt = ret_opt - tax_cost_opt
    sharpe_opt = (adjusted_ret_opt - risk_free_rate) / std_opt

    print("\nWITH LTCG tax cost penalty:")
    for ticker, w in zip(tickers, weights_opt):
        print(f"  {ticker}: {w:.2%}")

    print(f"\nExpected Return (before tax): {ret_opt:.2%}")
    print(f"Tax Cost (LTCG on sales): {tax_cost_opt:.2%}")
    print(f"Adjusted Return (after tax): {adjusted_ret_opt:.2%}")
    print(f"Volatility: {std_opt:.2%}")
    print(f"Sharpe Ratio (after tax): {sharpe_opt:.4f}")
else:
    print("Optimization failed.")

result_no_cost = minimize(
    neg_sharpe_without_cost,
    initial_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

if result_no_cost.success:
    weights_no_cost = result_no_cost.x
    ret_no_cost, std_no_cost = portfolio_performance(weights_no_cost)
    sharpe_no_cost = (ret_no_cost - risk_free_rate) / std_no_cost

    print("\nWITHOUT LTCG tax cost penalty:")
    for ticker, w in zip(tickers, weights_no_cost):
        print(f"  {ticker}: {w:.2%}")

    print(f"Expected Return: {ret_no_cost:.2%}")
    print(f"Volatility: {std_no_cost:.2%}")
    print(f"Sharpe Ratio: {sharpe_no_cost:.4f}")
else:
    print("Optimization without cost penalty failed.")

# Compute return, volatility, and Sharpe ratio for current portfolio (no sales)
ret_current, std_current = portfolio_performance(current_weights)
sharpe_current = (ret_current - risk_free_rate) / std_current

print(f"\nOriginal Portfolio (No Sales):")
print(f"Expected Return: {ret_current:.2%}")
print(f"Volatility: {std_current:.2%}")
print(f"Sharpe Ratio: {sharpe_current:.4f}")