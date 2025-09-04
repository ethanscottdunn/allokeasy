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

# --- Download price data ---
print("Downloading market data...")
try:
    data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    current_prices = data.iloc[-1]
    print("Data download successful!")
except Exception as e:
    print(f"Error downloading data: {e}")
    print("Using fallback data...")
    # Fallback data for testing
    mean_returns = pd.Series([0.12, 0.15, 0.10, 0.08, 0.25], index=tickers)
    cov_matrix = pd.DataFrame(np.eye(5) * 0.04, index=tickers, columns=tickers)
    current_prices = pd.Series([180, 380, 140, 150, 200], index=tickers)

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

def portfolio_performance(weights):
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
                lots_to_sell.append(Lot(quantity_to_sell, quantity_to_sell / (lot.quantity) * lot.cost_basis))
                lots_to_keep.append(Lot(lot.quantity - quantity_to_sell, (lot.quantity - quantity_to_sell) / lot.quantity * lot.cost_basis))
                realized_gains += quantity_to_sell * current_price - (quantity_to_sell / lot.quantity) * lot.cost_basis
            else:
                lots_to_keep.append(lot)
                realized_gains += lot.quantity * current_price - lot.cost_basis

        lots_to_sell_by_ticker[ticker] = lots_to_sell
        lots_to_keep_by_ticker[ticker] = lots_to_keep
        total_realized_gains += realized_gains
    
    total_tax = get_ltcg_tax(total_realized_gains, income, filing_status)
    return total_tax, lots_to_sell_by_ticker, lots_to_keep_by_ticker

def neg_sharpe_without_cost(weights):
    """Optimization function without tax consideration"""
    ret, std = portfolio_performance(weights)
    sharpe = (ret - risk_free_rate) / std
    return -sharpe

def neg_sharpe_with_lot_based_tax(weights):
    """Optimization function with lot-based tax consideration"""
    ret, std = portfolio_performance(weights)
    tax_cost, _, _ = calculate_lot_based_tax(lots_by_ticker, weights, current_prices, income, filing_status)
    adjusted_ret = ret - (tax_cost / total_portfolio_value)
    sharpe = (adjusted_ret - risk_free_rate) / std
    return -sharpe

# --- Create sample lots and calculate current portfolio ---
print("Creating lots by tickers from CSV")
# lots = create_sample_lots()
lots_by_ticker = parse_cost_basis_vanguard_csv()
total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
current_weights = np.array([sum(lot.quantity * current_prices[ticker] for lot in lots_by_ticker[ticker]) / total_portfolio_value for ticker in tickers])

print(f"\n=== PORTFOLIO OPTIMIZATION WITH LOT-BASED TAX CALCULATIONS ===")
print(f"User Income: ${income:,}")
print(f"Filing Status: {filing_status}")
print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")

# --- Current Portfolio Analysis ---
ret_current, std_current = portfolio_performance(current_weights)
sharpe_current = (ret_current - risk_free_rate) / std_current

print(f"\n=== CURRENT PORTFOLIO ===")
for i, ticker in enumerate(tickers):
    ticker_value = sum(lot.quantity * current_prices[ticker] for lot in lots_by_ticker[ticker])
    print(f"  {ticker}: {current_weights[i]:.2%} (${ticker_value:,.2f})")

print(f"Expected Return: {ret_current:.2%}")
print(f"Volatility: {std_current:.2%}")
print(f"Sharpe Ratio: {sharpe_current:.4f}")

# --- 1. Optimal Portfolio (No Tax Consideration) ---
print(f"\n=== 1. OPTIMAL PORTFOLIO (NO TAX CONSIDERATION) ===")
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in tickers)

result_optimal = minimize(
    neg_sharpe_without_cost,
    current_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

if result_optimal.success:
    weights_optimal = result_optimal.x
    ret_optimal, std_optimal = portfolio_performance(weights_optimal)
    sharpe_optimal = (ret_optimal - risk_free_rate) / std_optimal
    
    print("Optimal Weights:")
    for ticker, w in zip(tickers, weights_optimal):
        print(f"  {ticker}: {w:.2%}")
    
    print(f"Expected Return: {ret_optimal:.2%}")
    print(f"Volatility: {std_optimal:.2%}")
    print(f"Sharpe Ratio: {sharpe_optimal:.4f}")
else:
    print("Optimization failed.")

# --- 2. Tax Calculation for Optimal Portfolio ---
print(f"\n=== 2. TAX CALCULATION FOR OPTIMAL PORTFOLIO ===")
if result_optimal.success:
    tax_cost, lots_to_sell, lots_to_keep = calculate_lot_based_tax(lots_by_ticker, weights_optimal, current_prices, income, filing_status)
    
    print(f"Total Tax Cost: ${tax_cost:,.2f}")
    print(f"Tax Cost as % of Portfolio: {tax_cost/total_portfolio_value:.2%}")
    
    print("\nLots to Sell (MinTax Method):")
    for ticker in lots_to_sell:
        for lot_sale in lots_to_sell[ticker]:
            print(f"  {ticker}: {lot_sale.quantity:.0f} shares at ${lot_sale.cost_basis:.2f} cost basis")
    
# --- 3. Optimal Portfolio with Tax Consideration ---
print(f"\n=== 3. OPTIMAL PORTFOLIO WITH TAX CONSIDERATION ===")
result_tax_optimal = minimize(
    neg_sharpe_with_lot_based_tax,
    current_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

if result_tax_optimal.success:
    weights_tax_optimal = result_tax_optimal.x
    ret_tax_optimal, std_tax_optimal = portfolio_performance(weights_tax_optimal)
    tax_cost_tax_optimal, lots_to_sell_tax_optimal, lots_to_keep_tax_optimal = calculate_lot_based_tax(
        lots_by_ticker, weights_tax_optimal, current_prices, income, filing_status
    )
    adjusted_ret_tax_optimal = ret_tax_optimal - (tax_cost_tax_optimal / total_portfolio_value)
    sharpe_tax_optimal = (adjusted_ret_tax_optimal - risk_free_rate) / std_tax_optimal
    
    print("Tax-Aware Optimal Weights:")
    for ticker, w in zip(tickers, weights_tax_optimal):
        print(f"  {ticker}: {w:.2%}")
    
    print(f"Expected Return (before tax): {ret_tax_optimal:.2%}")
    print(f"Tax Cost: ${tax_cost_tax_optimal:,.2f} ({tax_cost_tax_optimal/total_portfolio_value:.2%})")
    print(f"Adjusted Return (after tax): {adjusted_ret_tax_optimal:.2%}")
    print(f"Volatility: {std_tax_optimal:.2%}")
    print(f"Sharpe Ratio (after tax): {sharpe_tax_optimal:.4f}")
    
    print(f"\nTax-Aware Lots to Sell:")
    for ticker in lots_to_sell_tax_optimal:
        for lot_sale in lots_to_sell_tax_optimal[ticker]:
            print(f"  {ticker}: {lot_sale.quantity:.2f} shares")
    
    print(f"\nTotal Tax Cost (Tax-Aware): ${tax_cost_tax_optimal:,.2f}")
else:
    print("Tax-aware optimization failed.")

# --- Summary Comparison ---
print(f"\n=== SUMMARY COMPARISON ===")
print(f"{'Metric':<25} {'Current':<12} {'Optimal (No Tax)':<18} {'Optimal (Tax-Aware)':<20}")
print("-" * 75)
print(f"{'Expected Return':<25} {ret_current:<12.2%} {ret_optimal:<18.2%} {ret_tax_optimal:<20.2%}")
print(f"{'Volatility':<25} {std_current:<12.2%} {std_optimal:<18.2%} {std_tax_optimal:<20.2%}")
print(f"{'Sharpe Ratio':<25} {sharpe_current:<12.4f} {sharpe_optimal:<18.4f} {sharpe_tax_optimal:<20.4f}")

if result_optimal.success and result_tax_optimal.success:
    print(f"{'Tax Cost':<25} {'N/A':<12} {tax_cost:<18,.0f} {tax_cost_tax_optimal:<20,.0f}")
    print(f"{'Tax Cost %':<25} {'N/A':<12} {tax_cost/total_portfolio_value:<18.2%} {tax_cost_tax_optimal/total_portfolio_value:<20.2%}")

print(f"\nNote: Tax calculations use MinTax method.")
print(f"Modify income and filing_status variables to match your tax situation.")