import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import random
from csv_parser.vanguard import Lot, parse_cost_basis_vanguard_csv

# --- Setup ---
# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
tickers = ['VASGX', 'VEMAX', 'VFIAX', 'VSIAX', 'VGPMX', 'VSGAX', 'VTSAX']
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')  # Use current date instead of 2025
risk_free_rate = 0.02
ltcg_single_first_breakpoint = 44625
ltcg_single_second_breakpoint = 492300
ltcg_married_first_breakpoint = 89250
ltcg_married_second_breakpoint = 553850
ltcg_zero_breakpoint_tax_rate = 0.00
ltcg_first_breakpoint_tax_rate = 0.15
ltcg_second_breakpoint_tax_rate = 0.20

income = 150000 
filing_status = 'single'

def get_yfinance_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        current_prices = data.iloc[-1]
        return mean_returns, cov_matrix, current_prices
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise Exception('failed to download data, check internet connection')

def get_ltcg_tax_by_breakpoints(realized_gains, income, ltcg_first_breakpoint, ltcg_second_breakpoint):
    ltcg_tax = 0
    available_bracket = max(ltcg_first_breakpoint - income, 0)
    used_bracket = min(available_bracket, realized_gains)
    ltcg_tax += used_bracket * ltcg_zero_breakpoint_tax_rate # technically 0 but just for consistency
    realized_gains -= used_bracket
    available_bracket = max(ltcg_second_breakpoint - max(ltcg_first_breakpoint, income), 0)
    used_bracket = min(available_bracket, realized_gains)
    ltcg_tax += used_bracket * ltcg_first_breakpoint_tax_rate
    realized_gains -= used_bracket
    ltcg_tax += realized_gains * ltcg_second_breakpoint_tax_rate
    return ltcg_tax

def get_ltcg_tax(realized_gains, income, filing_status):
    ltcg_tax = 0
    if filing_status == 'single':
        return get_ltcg_tax_by_breakpoints(realized_gains, income, ltcg_single_first_breakpoint, ltcg_single_second_breakpoint)
    elif filing_status == 'married':
        return get_ltcg_tax_by_breakpoints(realized_gains, income, ltcg_married_first_breakpoint, ltcg_married_second_breakpoint)
    else:
        raise Error(f'{filing_status=} is not supported')

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(weights.T @ cov_matrix.values @ weights)
    return ret, std

def calculate_lot_based_tax(lots_by_ticker, target_weights, current_prices, income, filing_status, tickers):

    total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
    target_values = {ticker: target_weights[i] * total_portfolio_value for i, ticker in enumerate(tickers)}
    
    total_realized_gains = 0
    lots_to_sell_by_ticker = {}
    lots_to_keep_by_ticker = {}
    
    for ticker in tickers:
        current_value = sum(lot.quantity * current_prices[ticker] for lot in lots_by_ticker[ticker])
        target_value = target_values[ticker]
        lots = lots_by_ticker[ticker]
        current_price = current_prices[ticker]
        quantity_to_sell = (current_value - target_value) / current_price
        realized_gains = 0
        lots_to_sell = []
        lots_to_keep = []

        for lot in lots:
            if quantity_to_sell <= 0:
                lots_to_keep.append(lot)
            elif quantity_to_sell < lot.quantity:
                lots_to_sell.append(Lot(quantity_to_sell, quantity_to_sell / (lot.quantity) * lot.cost_basis, lot.market_value))
                lots_to_keep.append(Lot(lot.quantity - quantity_to_sell, (lot.quantity - quantity_to_sell) / lot.quantity * lot.cost_basis, lot.market_value))
                realized_gains += quantity_to_sell * current_price - (quantity_to_sell / lot.quantity) * lot.cost_basis
            else:
                lots_to_keep.append(lot)
                realized_gains += lot.quantity * current_price - lot.cost_basis

        lots_to_sell_by_ticker[ticker] = lots_to_sell
        lots_to_keep_by_ticker[ticker] = lots_to_keep
        total_realized_gains += realized_gains
    
    total_tax = get_ltcg_tax(total_realized_gains, income, filing_status)
    return total_tax, lots_to_sell_by_ticker, lots_to_keep_by_ticker

def compare_contrast_portfolios(start_date, end_date, risk_free_rate, income, filing_status):
    lots_by_ticker = parse_cost_basis_vanguard_csv()
    tickers = sorted(lots_by_ticker)
    mean_returns, cov_matrix, current_prices = get_yfinance_data(tickers, start_date, end_date)
    return [
        original_portfolio(tickers, lots_by_ticker, mean_returns, cov_matrix, current_prices, risk_free_rate),
        optimized_portfolio(tickers, lots_by_ticker, mean_returns, cov_matrix, current_prices, risk_free_rate, income, filing_status),
        tax_optimized_portfolio(tickers, lots_by_ticker, mean_returns, cov_matrix, current_prices, risk_free_rate, income, filing_status)
    ]

def original_portfolio(tickers, lots_by_ticker, mean_returns, cov_matrix, current_prices, risk_free_rate):
    market_value_by_ticker = {ticker: sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items()}
    total_portfolio_value = sum(market_value for market_value in market_value_by_ticker.values())
    current_weights = np.array([sum(lot.quantity * current_prices[ticker] for lot in lots_by_ticker[ticker]) / total_portfolio_value for ticker in tickers])

    ret, std = portfolio_performance(current_weights, mean_returns, cov_matrix)
    sharpe = (ret - risk_free_rate) / std

    return {'value': total_portfolio_value, 'ret': ret, 'std': std, 'sharpe': sharpe, 'taxes_paid': 0.0, 'portfolio': [(ticker, current_weights[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]}

def optimized_portfolio(tickers, lots_by_ticker, mean_returns, cov_matrix, current_prices, risk_free_rate, income, filing_status):
    total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
    current_weights = np.array([sum(lot.quantity * current_prices[ticker] for lot in lots_by_ticker[ticker]) / total_portfolio_value for ticker in tickers])

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in tickers)

    neg_sharpe_without_cost = lambda w: -((np.dot(w, mean_returns) - risk_free_rate) / np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))

    result_optimal = minimize(
        neg_sharpe_without_cost,
        current_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result_optimal.success:
        weights_optimal = result_optimal.x
        ret_optimal, std_optimal = portfolio_performance(weights_optimal, mean_returns, cov_matrix)
        tax_cost_optimal, _, _ = calculate_lot_based_tax(lots_by_ticker, weights_optimal, current_prices, income, filing_status, tickers)
        sharpe_optimal = (ret_optimal - risk_free_rate) / std_optimal
        ret_optimal_post_tax = ret_optimal - tax_cost_optimal / total_portfolio_value
        sharpe_optimal_post_tax = (ret_optimal_post_tax - risk_free_rate) / std_optimal
        return {'value': total_portfolio_value, 'ret': {'pre_tax': ret_optimal, 'post_tax': ret_optimal_post_tax}, 'std': std_optimal, 'sharpe': {'pre_tax': sharpe_optimal, 'post_tax': sharpe_optimal_post_tax}, 'taxes_paid': tax_cost_optimal, 'portfolio': [(ticker, weights_optimal[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]}
    else:
        print("Optimization failed.")

def tax_optimized_portfolio(tickers, lots_by_ticker, mean_returns, cov_matrix, current_prices, risk_free_rate, income, filing_status):
    total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
    current_weights = np.array([sum(lot.quantity * current_prices[ticker] for lot in lots_by_ticker[ticker]) / total_portfolio_value for ticker in tickers])

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in tickers)

    neg_sharpe_with_lot_based_tax = lambda w: -(((np.dot(w, mean_returns) - (calculate_lot_based_tax(lots_by_ticker, w, current_prices, income, filing_status, tickers)[0] / total_portfolio_value) - risk_free_rate)
    / np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))))

    result_tax_optimal = minimize(
        neg_sharpe_with_lot_based_tax,
        current_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result_tax_optimal.success:
        weights_tax_optimal = result_tax_optimal.x
        ret_tax_optimal, std_tax_optimal = portfolio_performance(weights_tax_optimal, mean_returns, cov_matrix)
        tax_cost_tax_optimal, lots_to_sell_tax_optimal, lots_to_keep_tax_optimal = calculate_lot_based_tax(
            lots_by_ticker, weights_tax_optimal, current_prices, income, filing_status, tickers
        )
        adjusted_ret_tax_optimal = ret_tax_optimal - (tax_cost_tax_optimal / total_portfolio_value)
        sharpe_tax_optimal = (adjusted_ret_tax_optimal - risk_free_rate) / std_tax_optimal
        return {'value': total_portfolio_value, 'ret': adjusted_ret_tax_optimal, 'std': std_tax_optimal, 'sharpe': sharpe_tax_optimal, 'taxes_paid': tax_cost_tax_optimal, 'portfolio': [(ticker, weights_tax_optimal[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]}
    else:
        print("Tax-aware optimization failed.")

def plot_investments(portfolios, years=10):
    """
    portfolios: list of dicts with keys:
        - start_value (float)
        - annual_return (float, expected return per year, e.g. 0.06 for 6%)
        - annual_std (float, standard deviation per year, e.g. 0.15 for 15%)
        - label (str)
        - color (str)
    years: int, max horizon to plot
    """

    t = np.arange(0, years + 1)

    plt.figure(figsize=(10, 6))

    growth_curves = {}

    for p in portfolios:
        # Expected compounded growth
        expected = p["start_value"] * (1 + p["annual_return"]) ** t
        growth_curves[p["label"]] = expected

        # Plot without shading
        plt.plot(t, expected, label=p["label"], color=p["color"])

    # --- Find intersection between "Current" and "Tax-Optimized" ---
    if "Current" in growth_curves and "Tax-Optimized" in growth_curves:
        current = growth_curves["Current"]
        tax_opt = growth_curves["Tax-Optimized"]

        # Look for sign change in the difference
        diff = tax_opt - current
        for i in range(1, len(t)):
            if diff[i-1] * diff[i] <= 0:  # sign change → intersection
                # Linear interpolation for better accuracy
                x0, x1 = t[i-1], t[i]
                y0, y1 = diff[i-1], diff[i]
                intersect_x = x0 - y0 * (x1 - x0) / (y1 - y0)
                intersect_y = np.interp(intersect_x, t, tax_opt)

                # Add vertical line & label
                plt.axvline(intersect_x, color="gray", linestyle="--", alpha=0.7)
                plt.text(intersect_x, plt.ylim()[1]*0.95, f"{intersect_x:.2f} yrs",
                         rotation=90, va="top", ha="right", fontsize=9, color="gray")
                break  # stop at first intersection

    plt.title("Investment Growth Comparison", fontsize=14)
    plt.xlabel("Years")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

if __name__ == "__main__":
    original, optimized, tax_optimized = compare_contrast_portfolios('2020-01-01', '2025-01-01', 0.02, 150000, 'single')
    lots_by_ticker = parse_cost_basis_vanguard_csv()
    tickers = sorted(lots_by_ticker)
    mean_returns, cov_matrix, current_prices = get_yfinance_data(tickers, start_date, end_date)
    total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
    portfolios = [
        {"start_value": original['value'], "annual_return": original['ret'], "annual_std": original['std'], "label": "Current", "color": "blue"},
        {"start_value": optimized['value'], "annual_return": optimized['ret']['pre_tax'], "annual_std": optimized['std'], "label": "Optimized Pre-Tax", "color": "red"},
        {"start_value": optimized['value'] - optimized['taxes_paid'], "annual_return": optimized['ret']['post_tax'], "annual_std": optimized['std'], "label": "Optimized Post-Tax", "color": "orange"},
        {"start_value": tax_optimized['value'] - tax_optimized['taxes_paid'], "annual_return": tax_optimized['ret'], "annual_std": tax_optimized['std'], "label": "Tax-Optimized", "color": "green"},
    ]
    plot_investments(portfolios, years=3)

