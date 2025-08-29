import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import random

# --- Setup ---
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')  # Use current date instead of 2025
risk_free_rate = 0.02

# --- LTCG Tax Rates by Income Level (2024 rates) ---
ltcg_tax_brackets = {
    '0%': 0,      # $0 - $44,625 (single), $0 - $89,250 (married)
    '15%': 0.15,  # $44,626 - $492,300 (single), $89,251 - $553,850 (married)
    '20%': 0.20   # Over $492,300 (single), Over $553,850 (married)
}

# User's income level (modify as needed)
user_income = 150000  # $150k annual income
user_filing_status = 'single'  # 'single' or 'married'

def get_ltcg_tax_rate(income, filing_status='single'):
    """Determine LTCG tax rate based on income and filing status"""
    if filing_status == 'single':
        if income <= 44625:
            return 0.0
        elif income <= 492300:
            return 0.15
        else:
            return 0.20
    else:  # married
        if income <= 89250:
            return 0.0
        elif income <= 553850:
            return 0.15
        else:
            return 0.20

# --- Current Portfolio with Lot Information ---
class Lot:
    def __init__(self, ticker, shares, cost_basis, purchase_date):
        self.ticker = ticker
        self.shares = shares
        self.cost_basis = cost_basis  # per share
        self.purchase_date = purchase_date

# Simulate current holdings with lots (you can modify these based on your actual holdings)
def create_sample_lots():
    lots = []
    current_prices = {}
    
    # Get current prices for calculations
    try:
        current_data = yf.download(tickers, start=datetime.now() - timedelta(days=5), end=datetime.now())['Close']
        current_prices = current_data.iloc[-1]
    except:
        # Fallback prices if download fails
        current_prices = {'AAPL': 180, 'MSFT': 380, 'GOOGL': 140, 'AMZN': 150, 'TSLA': 200}
    
    # Create sample lots for each ticker
    for ticker in tickers:
        # Simulate multiple lots with different cost bases
        num_lots = random.randint(2, 4)
        total_shares = 1000  # Total shares per ticker
        
        for i in range(num_lots):
            shares = total_shares // num_lots
            if i == num_lots - 1:  # Last lot gets remaining shares
                shares = total_shares - sum(lot.shares for lot in lots if lot.ticker == ticker)
            
            # Simulate different purchase dates and cost bases
            purchase_date = datetime.now() - timedelta(days=random.randint(30, 1000))
            cost_basis = current_prices[ticker] * random.uniform(0.5, 1.5)  # 50% to 150% of current price
            
            lots.append(Lot(ticker, shares, cost_basis, purchase_date))
    
    return lots

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

def portfolio_performance(weights):
    """Calculate portfolio return and volatility"""
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(weights.T @ cov_matrix.values @ weights)
    return ret, std

def calculate_lot_based_tax(lots, target_weights, current_prices, tax_rate):
    """
    Calculate tax cost using lot-based selling (FIFO method)
    Returns: total tax cost, lots to sell, and remaining lots
    """
    total_portfolio_value = sum(lot.shares * current_prices[lot.ticker] for lot in lots)
    target_values = {ticker: target_weights[i] * total_portfolio_value for i, ticker in enumerate(tickers)}
    
    # Group lots by ticker
    lots_by_ticker = {}
    for lot in lots:
        if lot.ticker not in lots_by_ticker:
            lots_by_ticker[lot.ticker] = []
        lots_by_ticker[lot.ticker].append(lot)
    
    # Sort lots by purchase date (FIFO)
    for ticker in lots_by_ticker:
        lots_by_ticker[ticker].sort(key=lambda x: x.purchase_date)
    
    total_tax = 0
    lots_to_sell = []
    remaining_lots = []
    
    for ticker in tickers:
        current_value = sum(lot.shares * current_prices[ticker] for lot in lots_by_ticker.get(ticker, []))
        target_value = target_values[ticker]
        
        if current_value > target_value:
            # Need to sell some shares
            shares_to_sell = (current_value - target_value) / current_prices[ticker]
            shares_sold = 0
            ticker_tax = 0
            
            for lot in lots_by_ticker[ticker]:
                if shares_sold >= shares_to_sell:
                    remaining_lots.append(lot)
                    continue
                
                shares_available = lot.shares
                shares_from_this_lot = min(shares_available, shares_to_sell - shares_sold)
                
                if shares_from_this_lot > 0:
                    # Calculate gain/loss on this lot
                    gain_per_share = current_prices[ticker] - lot.cost_basis
                    if gain_per_share > 0:  # Only tax gains
                        lot_tax = shares_from_this_lot * gain_per_share * tax_rate
                        ticker_tax += lot_tax
                    
                    shares_sold += shares_from_this_lot
                    
                    if shares_from_this_lot < shares_available:
                        # Partial lot sale
                        remaining_shares = shares_available - shares_from_this_lot
                        remaining_lots.append(Lot(ticker, remaining_shares, lot.cost_basis, lot.purchase_date))
                    
                    lots_to_sell.append({
                        'ticker': ticker,
                        'shares': shares_from_this_lot,
                        'cost_basis': lot.cost_basis,
                        'sale_price': current_prices[ticker],
                        'gain': gain_per_share * shares_from_this_lot,
                        'tax': lot_tax if gain_per_share > 0 else 0
                    })
            
            total_tax += ticker_tax
        else:
            # Keep all lots for this ticker
            remaining_lots.extend(lots_by_ticker.get(ticker, []))
    
    return total_tax, lots_to_sell, remaining_lots

def neg_sharpe_without_cost(weights):
    """Optimization function without tax consideration"""
    ret, std = portfolio_performance(weights)
    sharpe = (ret - risk_free_rate) / std
    return -sharpe

def neg_sharpe_with_lot_based_tax(weights):
    """Optimization function with lot-based tax consideration"""
    ret, std = portfolio_performance(weights)
    tax_cost, _, _ = calculate_lot_based_tax(lots, weights, current_prices, ltcg_tax_rate)
    adjusted_ret = ret - (tax_cost / total_portfolio_value)
    sharpe = (adjusted_ret - risk_free_rate) / std
    return -sharpe

# --- Create sample lots and calculate current portfolio ---
print("Creating sample portfolio lots...")
lots = create_sample_lots()
total_portfolio_value = sum(lot.shares * current_prices[lot.ticker] for lot in lots)
current_weights = np.array([sum(lot.shares * current_prices[lot.ticker] for lot in lots if lot.ticker == ticker) / total_portfolio_value for ticker in tickers])

# Get user's LTCG tax rate
ltcg_tax_rate = get_ltcg_tax_rate(user_income, user_filing_status)

print(f"\n=== PORTFOLIO OPTIMIZATION WITH LOT-BASED TAX CALCULATIONS ===")
print(f"User Income: ${user_income:,}")
print(f"Filing Status: {user_filing_status}")
print(f"LTCG Tax Rate: {ltcg_tax_rate:.1%}")
print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")

# --- Current Portfolio Analysis ---
ret_current, std_current = portfolio_performance(current_weights)
sharpe_current = (ret_current - risk_free_rate) / std_current

print(f"\n=== CURRENT PORTFOLIO ===")
for i, ticker in enumerate(tickers):
    ticker_value = sum(lot.shares * current_prices[ticker] for lot in lots if lot.ticker == ticker)
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
    tax_cost, lots_to_sell, remaining_lots = calculate_lot_based_tax(lots, weights_optimal, current_prices, ltcg_tax_rate)
    
    print(f"Total Tax Cost: ${tax_cost:,.2f}")
    print(f"Tax Cost as % of Portfolio: {tax_cost/total_portfolio_value:.2%}")
    
    print("\nLots to Sell (FIFO Method):")
    for lot_sale in lots_to_sell:
        print(f"  {lot_sale['ticker']}: {lot_sale['shares']:.0f} shares at ${lot_sale['cost_basis']:.2f} cost basis")
        print(f"    Sale Price: ${lot_sale['sale_price']:.2f}, Gain: ${lot_sale['gain']:.2f}, Tax: ${lot_sale['tax']:.2f}")
    
    print(f"\nRemaining Lots: {len(remaining_lots)} lots")

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
    tax_cost_tax_optimal, lots_to_sell_tax_optimal, remaining_lots_tax_optimal = calculate_lot_based_tax(
        lots, weights_tax_optimal, current_prices, ltcg_tax_rate
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
    for lot_sale in lots_to_sell_tax_optimal:
        print(f"  {lot_sale['ticker']}: {lot_sale['shares']:.0f} shares, Tax: ${lot_sale['tax']:.2f}")
    
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

print(f"\nNote: Tax calculations use FIFO method and current LTCG rates for {user_filing_status} filers.")
print(f"Modify user_income and user_filing_status variables to match your tax situation.")