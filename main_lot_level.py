import yfinance as yf
import numpy as np
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

def calculate_lot_based_tax(lots_by_ticker, target_weights, current_prices, income, filing_status):

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

# def neg_sharpe_without_cost(weights):
#     """Optimization function without tax consideration"""
#     ret, std = portfolio_performance(weights)
#     sharpe = (ret - risk_free_rate) / std
#     return -sharpe

# def neg_sharpe_with_lot_based_tax(weights):
#     """Optimization function with lot-based tax consideration"""
#     ret, std = portfolio_performance(weights)
#     tax_cost, _, _ = calculate_lot_based_tax(lots_by_ticker, weights, current_prices, income, filing_status)
#     adjusted_ret = ret - (tax_cost / total_portfolio_value)
#     sharpe = (adjusted_ret - risk_free_rate) / std
#     return -sharpe

def compare_contrast_portfolios(csv_file_path, start_date, end_date, risk_free_rate, income, filing_status):
    lots_by_ticker = parse_cost_basis_vanguard_csv(csv_file_path)
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

    print({'ret': ret, 'std': std, 'sharpe': sharpe, 'taxes_paid': 0.0, 'portfolio': [(ticker, current_weights[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]})
    return {'ret': ret, 'std': std, 'sharpe': sharpe, 'taxes_paid': 0.0, 'portfolio': [(ticker, current_weights[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]}

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
        tax_cost_optimal, _, _ = calculate_lot_based_tax(lots_by_ticker, weights_optimal, current_prices, income, filing_status)
        ret_optimal = ret_optimal - tax_cost_optimal / total_portfolio_value # adjusting returns for the taxes you do have to pay
        sharpe_optimal = (ret_optimal - risk_free_rate) / std_optimal
        print({'ret': ret_optimal, 'std': std_optimal, 'sharpe': sharpe_optimal, 'taxes_paid': tax_cost_optimal, 'portfolio': [(ticker, weights_optimal[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]})
        return {'ret': ret_optimal, 'std': std_optimal, 'sharpe': sharpe_optimal, 'taxes_paid': tax_cost_optimal, 'portfolio': [(ticker, weights_optimal[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]}
    else:
        print("Optimization failed.")

def tax_optimized_portfolio(tickers, lots_by_ticker, mean_returns, cov_matrix, current_prices, risk_free_rate, income, filing_status):
    total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
    current_weights = np.array([sum(lot.quantity * current_prices[ticker] for lot in lots_by_ticker[ticker]) / total_portfolio_value for ticker in tickers])

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in tickers)

    neg_sharpe_with_lot_based_tax = lambda w: -(((np.dot(w, mean_returns) - (calculate_lot_based_tax(lots_by_ticker, w, current_prices, income, filing_status)[0] / total_portfolio_value) - risk_free_rate)
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
            lots_by_ticker, weights_tax_optimal, current_prices, income, filing_status
        )
        adjusted_ret_tax_optimal = ret_tax_optimal - (tax_cost_tax_optimal / total_portfolio_value)
        sharpe_tax_optimal = (adjusted_ret_tax_optimal - risk_free_rate) / std_tax_optimal
        print({'ret': adjusted_ret_tax_optimal, 'std': std_tax_optimal, 'sharpe': sharpe_tax_optimal, 'taxes_paid': tax_cost_tax_optimal, 'portfolio': [(ticker, weights_tax_optimal[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]})
        return {'ret': adjusted_ret_tax_optimal, 'std': std_tax_optimal, 'sharpe': sharpe_tax_optimal, 'taxes_paid': tax_cost_tax_optimal, 'portfolio': [(ticker, weights_tax_optimal[i] * total_portfolio_value) for i, ticker in enumerate(tickers)]}
    else:
        print("Tax-aware optimization failed.")

if __name__ == "__main__":
    compare_contrast_portfolios('2020-01-01', '2025-01-01', 0.02, 150000, 'single')

