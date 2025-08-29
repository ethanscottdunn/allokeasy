import csv

csv_file_path = 'cost_basis_vanguard_test_0.csv'
symbol_column = 'Symbol/CUSIP'
total_cost_column = 'Total cost'
quantity_column = 'Quantity'

class Lot:
    def __init__(self, quantity, cost_basis):
        self.quantity = quantity
        self.cost_basis = cost_basis

def parse_cost_basis_vanguard_csv():

	lots_by_ticker: Dict[str, list[Lot]] = {}

	with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
		next(file)
		next(file)
		reader = csv.DictReader(file)
		headers = next(reader)
		for row in reader:
		    ticker = row[symbol_column]
		    quantity = row[quantity_column]
		    cost_basis = row[total_cost_column]
		    if ticker not in lots_by_ticker:
		    	lots_by_ticker[ticker] = []
		    lots_by_ticker[ticker].append(Lot(quantity, cost_basis))
		for lots in lots_by_ticker:
			lots.sort(lambda x: x.cost_basis/x.quantity)
			print(f"{ticker=}, {lots=}")

	return lots_by_ticker
