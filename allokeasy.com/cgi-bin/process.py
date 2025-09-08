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
from utils.main_lot_level import compare_contrast_portfolios

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
