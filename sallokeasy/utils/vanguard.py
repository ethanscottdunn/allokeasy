import csv

csv_file_path = 'csv_parser/cost_basis_vanguard_test_0.csv'
market_value_column_prefix = 'Market value as of'
symbol_column = 'Symbol/CUSIP'
total_cost_column = 'Total cost'
quantity_column = 'Quantity'

class Lot:
    def __init__(self, quantity, cost_basis, market_value):
        self.quantity = quantity
        self.cost_basis = cost_basis
        self.market_value = market_value

    def __repr__(self):
    	return f'{self.quantity=}, {self.cost_basis=}, {self.market_value=}'

def parse_cost_basis_vanguard_csv(csv_file_path=csv_file_path):

	lots_by_ticker: Dict[str, list[Lot]] = {}
	file = csv_file_path
	#with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
	next(file)
	next(file)
	reader = csv.DictReader(file)
	headers = next(reader)
	market_value_column = ''
	for header in headers:
		if header.startswith(market_value_column_prefix):
			market_value_column = header
	if not market_value_column:
		raise Exception('could not find a column that starts with prefix {market_value_column_prefix}')
	for row in reader:
		ticker = row[symbol_column]
		quantity = float(row[quantity_column])
		cost_basis = float(row[total_cost_column])
		market_value = float(row[market_value_column])
		if ticker not in lots_by_ticker:
			lots_by_ticker[ticker] = []
		lots_by_ticker[ticker].append(Lot(quantity, cost_basis, market_value))
	for lots in lots_by_ticker.values():
		lots.sort(key=lambda x: x.cost_basis/x.quantity)

	return lots_by_ticker

def parse_cost_basis_vanguard_csv_by_ticker(csv_file_path):

	lots_by_ticker = parse_cost_basis_vanguard_csv(csv_file_path)

	return [{'ticker': ticker, 'quantity': f"{sum(lot.quantity for lot in lots):.2f}", 'market_value': f"{(sum(lot.market_value for lot in lots)):.2f}"} for ticker, lots in lots_by_ticker.items()]
