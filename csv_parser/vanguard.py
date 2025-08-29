import csv

csv_file_path = 'cost_basis_vanguard_test_0.csv'
symbol_column = 'Symbol/CUSIP'
total_cost_column = 'Total cost'
market_value_column_prefix = 'Market value as of'

def parse_cost_basis_vanguard_csv():

	with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
		next(file)
		next(file)
		reader = csv.DictReader(file)
		headers = next(reader)
		print(f"{headers=}")
		market_value_columns = [col for col in headers if col.startswith(market_value_column_prefix)]
		assert(len(market_value_columns) == 1)
		market_value_column = market_value_columns[0]
		for row in reader:
		    symbol = row[symbol_column]
		    cost_basis = row[total_cost_column]
		    market_value = row[market_value_column]
		    print(f"{symbol=}, {cost_basis=}, {market_value=}")
