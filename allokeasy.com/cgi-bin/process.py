#!/usr/bin/env python3
import cgi, cgitb, json, sys, io, csv, os
from datetime import datetime

# --- Activate virtualenv ---
venv_path = os.path.expanduser("~/allokeasy.com/env")  # adjust path
activate_this = os.path.join(venv_path, "bin", "activate_this.py")

with open(activate_this) as f:
    exec(f.read(), {"__file__": activate_this})

#sys.path.append(os.path.dirname(__file__))  # make sure cgi-bin is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

#from utils import test
#from utils import main_lot_level
from utils.main_lot_level import compare_contrast_portfolios, plot_investments, get_yfinance_data
from utils.vanguard import parse_cost_basis_vanguard_csv

cgitb.enable()  # for debugging in browser

print("Content-Type: application/json\n")

form = cgi.FieldStorage()

# Read inputs
income = float(form.getfirst("income", "0"))
risk_free_rate = float(form.getfirst("risk_free_rate", "0"))
status = form.getfirst("status", "")
start_date = form.getfirst("start_date", "")
end_date = form.getfirst("end_date", "")

# married status handling isn't fully implemented?
if status == "married_joint" or status =="married_separate":
    status = "married"

# Handle uploaded CSV
csvfile = form["csvfile"]
table_data = []
if csvfile.file:
    csv_bytes = csvfile.file.read()
    csv_text = csv_bytes.decode("utf-8")
    io_string = io.StringIO(csv_text)
    reader = csv.reader(io.StringIO(csv_text))
    table_data = list(reader)

#decoded_file = csv_file.read().decode("utf-8")
#io_string = io.StringIO(decoded_file)
csv_file = io_string
# ---- Compute the portfolios  ----
no_change, from_scratch, tax_optimized = compare_contrast_portfolios(csv_file, start_date, end_date, risk_free_rate, income, status)
csv_file = io.StringIO(csv_text)
lots_by_ticker = parse_cost_basis_vanguard_csv(csv_file)
tickers = sorted(lots_by_ticker)
mean_returns, cov_matrix, current_prices = get_yfinance_data(tickers, start_date, end_date)
total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
portfolios = [
    {"start_value": no_change['value'], "annual_return": no_change['ret'], "annual_std": no_change['std'], "label": "Current", "color": "blue"},
    {"start_value": from_scratch['value'], "annual_return": from_scratch['ret']['pre_tax'], "annual_std": from_scratch['std'], "label": "Optimized Pre-Tax", "color": "red"},
    {"start_value": from_scratch['value'] - from_scratch['taxes_paid'], "annual_return": from_scratch['ret']['post_tax'], "annual_std": from_scratch['std'], "label": "Optimized Post-Tax", "color": "orange"},
    {"start_value": tax_optimized['value'] - tax_optimized['taxes_paid'], "annual_return": tax_optimized['ret'], "annual_std": tax_optimized['std'], "label": "Tax-Optimized", "color": "green"},
]
plot_investments(portfolios, years=3)
# Ethan's code to call the plotting, which I'm replacing above with the variables for the website
# original, optimized, tax_optimized = compare_contrast_portfolios('2020-01-01', '2025-01-01', 0.02, 150000, 'single')
# lots_by_ticker = parse_cost_basis_vanguard_csv()
# tickers = sorted(lots_by_ticker)
# mean_returns, cov_matrix, current_prices = get_yfinance_data(tickers, start_date, end_date)
# total_portfolio_value = sum(sum(lot.quantity * current_prices[ticker] for lot in lots) for ticker, lots in lots_by_ticker.items())
# portfolios = [
#     {"start_value": original['value'], "annual_return": original['ret'], "annual_std": original['std'], "label": "Current", "color": "blue"},
#     {"start_value": optimized['value'], "annual_return": optimized['ret']['pre_tax'], "annual_std": optimized['std'], "label": "Optimized Pre-Tax", "color": "red"},
#     {"start_value": optimized['value'] - optimized['taxes_paid'], "annual_return": optimized['ret']['post_tax'], "annual_std": optimized['std'], "label": "Optimized Post-Tax", "color": "orange"},
#     {"start_value": tax_optimized['value'] - tax_optimized['taxes_paid'], "annual_return": tax_optimized['ret'], "annual_std": tax_optimized['std'], "label": "Tax-Optimized", "color": "green"},
# ]
# plot_investments(portfolios, years=3)


dict1 = {"A": "Value1", "B": "Value2"}
dict2 = {"C": "Value3", "D": "Value4"}
dict3 = {"E": "Value5", "F": "Value6"}
# For demo, just echoing table rows with 2 cols
table = [[row[0], row[1] if len(row) > 1 else ""] for row in table_data]

# Build response
#result = {
#    "dict1": dict1,
#    "dict2": dict2,
#    "dict3": dict3,
#    "table": table
#}

result = {
        "original": no_change,
        "optimized": from_scratch,
        "tax_optimized": tax_optimized,
}


print(json.dumps(result))
