import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Get data
def fetch_price_data(tickers, start='2020-01-01', end='2025-01-01'):
    data = yf.download(tickers, start=start, end=end)['Close']
    return data.dropna()

# 2. Portfolio statistics
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    ret, std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / std

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets], args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# 3. Generate Efficient Frontier
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        weights_record.append(weights)
        port_return, port_std = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (port_return - risk_free_rate) / port_std
        results[0,i] = port_std
        results[1,i] = port_return
        results[2,i] = sharpe

    return results, weights_record

# 4. Plotting
def plot_efficient_frontier(results, max_sharpe, max_sharpe_return, max_sharpe_std):
    plt.figure(figsize=(10,7))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe_std, max_sharpe_return, marker='*', color='r', s=200, label='Max Sharpe Ratio')
    plt.title('Efficient Frontier with Random Portfolios')
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Run the simulation
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    data = fetch_price_data(tickers)
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    results, _ = generate_random_portfolios(10000, mean_returns, cov_matrix)
    opt_result = optimize_portfolio(mean_returns, cov_matrix)
    max_sharpe_weights = opt_result.x
    max_sharpe_return, max_sharpe_std = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)

    plot_efficient_frontier(results, max_sharpe_weights, max_sharpe_return, max_sharpe_std)

    print("Optimal Weights for Max Sharpe Ratio Portfolio:")
    for ticker, weight in zip(tickers, max_sharpe_weights):
        print(f"{ticker}: {weight:.2%}")

