# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlloKeasy is a portfolio optimization tool that helps investors rebalance their portfolios while accounting for capital gains taxes. It computes three portfolio scenarios:
1. **Original**: Current portfolio performance
2. **Optimized**: Optimal portfolio ignoring taxes (shows theoretical best performance)
3. **Tax-Optimized**: Optimal portfolio accounting for capital gains taxes

The tool uses Modern Portfolio Theory (Sharpe ratio maximization) with scipy optimization and integrates lot-level tax calculations based on US long-term capital gains tax brackets.

## Architecture

### Core Calculation Flow

1. **CSV Parsing** (`csv_parser/vanguard.py`): Parses Vanguard cost-basis CSV files into `Lot` objects (quantity, cost basis, market value). Lots are sorted by cost-basis per share to enable MinTax selling strategy.

2. **Portfolio Optimization** (`main_lot_level.py` and `allokeasy.com/cgi-bin/utils/main_lot_level.py`):
   - Downloads historical price data from Yahoo Finance using yfinance
   - Calculates mean returns and covariance matrix (annualized)
   - Runs scipy.optimize.minimize with SLSQP method to maximize Sharpe ratio
   - For tax-optimized portfolio: objective function includes `calculate_lot_based_tax()` in the return calculation

3. **Tax Calculation** (`calculate_lot_based_tax()` in main_lot_level.py):
   - Determines which lots to sell based on MinTax strategy (sells lowest cost-basis lots first)
   - Calculates realized gains for each lot sold
   - Applies progressive LTCG tax rates (0%, 15%, 20%) based on income and filing status

4. **Results Visualization**: Generates matplotlib plot showing expected portfolio growth over time for all scenarios

### Dual Deployment Architecture

The codebase supports both **local execution** and **web deployment**:

- **Local scripts**: `main_lot_level.py`, `main.py`, `main_plot.py` in root directory
- **Web backend**: `allokeasy.com/cgi-bin/process.py` (CGI script that accepts CSV uploads and user inputs via HTML forms)
- **Shared utilities**: `allokeasy.com/cgi-bin/utils/` contains web versions that mirror local scripts with adjusted import paths

### Key Modules

- `csv_parser/vanguard.py`: CSV parsing and `Lot` class definition
- `main_lot_level.py`: Core optimization logic (local version)
- `allokeasy.com/cgi-bin/utils/main_lot_level.py`: Core optimization logic (web version, nearly identical)
- `allokeasy.com/cgi-bin/utils/vanguard.py`: Web version of CSV parser
- `allokeasy.com/cgi-bin/process.py`: CGI entry point for web interface

## Development Commands

### Running Local Scripts

```bash
# Run full optimization with test CSV data
python main_lot_level.py

# Run basic optimization example (simplified tickers)
python main.py

# Run plotting example
python main_plot.py
```

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

Required packages: yfinance, scipy, matplotlib, django (for web deployment)

### Testing CSV Parser

```bash
# Parse Vanguard CSV and print lot information
python csv_parser/cost_basis_parser.py
```

## Important Implementation Details

### Vanguard CSV Format

The parser expects specific columns:
- `Symbol/CUSIP`: Ticker symbol
- `Quantity`: Number of shares in lot
- `Total cost`: Cost basis for the lot
- `Market value as of [date]`: Current market value (column name includes date)

The CSV has 2 header rows that are skipped before reading the data.

### MinTax Lot Selling Strategy

When rebalancing, lots are sold in order of lowest cost-basis per share first. This minimizes taxes for a given dollar amount sold. The `calculate_lot_based_tax()` function implements this by:
1. Sorting lots by `cost_basis/quantity` (done in CSV parser)
2. Selling from sorted list until target value is reached
3. Tracking which lots are partially vs fully sold

### Tax Calculation Details

LTCG brackets (2024 values hardcoded):
- Single: $0-$44,625 (0%), $44,625-$492,300 (15%), $492,300+ (20%)
- Married: $0-$89,250 (0%), $89,250-$553,850 (15%), $553,850+ (20%)

Income fills brackets first, then capital gains fill remaining space in each bracket.

### Optimization Constraints

- Sum of weights must equal 1 (fully invested)
- All weights between 0 and 1 (no shorting, no leverage)
- Uses SLSQP optimizer (handles constraints well for this problem)
- Initial guess is current portfolio weights

### Web Interface Integration

`allokeasy.com/cgi-bin/process.py` accepts:
- `csvfile`: Uploaded Vanguard CSV
- `income`: User's annual income
- `risk_free_rate`: Risk-free rate for Sharpe calculation
- `status`: Filing status (single/married)
- `start_date`, `end_date`: Date range for historical data

Returns JSON with three portfolio dictionaries containing performance metrics.

## Common Pitfalls

1. **Import paths differ between local and web**: Local scripts use `from csv_parser.vanguard import ...`, web scripts use `from utils.vanguard import ...`

2. **CSV file path handling**: Local scripts hardcode path in `csv_parser/vanguard.py`, web version accepts file-like object

3. **Plotting in CGI context**: Web version saves plots to SVG file instead of using `plt.show()`

4. **Yahoo Finance API**: Can be rate-limited or temporarily unavailable. Error handling raises exception with message "failed to download data, check internet connection"

5. **Date handling**: `main_lot_level.py` uses `datetime.now()` for end_date, not hardcoded 2025 date

## Code Location Reference

- Portfolio optimization objective functions: `main_lot_level.py:148` (without tax), `main_lot_level.py:176` (with tax)
- Tax calculation: `main_lot_level.py:70-105`
- CSV parsing: `csv_parser/vanguard.py:18-44`
- Web CGI handler: `allokeasy.com/cgi-bin/process.py:51`
- Lot class definition: `csv_parser/vanguard.py:9-16`
