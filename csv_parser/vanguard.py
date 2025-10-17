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
	"""
	Parse Vanguard cost basis CSV with automatic header detection.
	Handles variable number of lines before the actual header row.
	"""
	lots_by_ticker: Dict[str, list[Lot]] = {}

	with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
		# Read all lines to find the header row
		file.seek(0)  # Reset file pointer to beginning
		lines = [line for line in file]

		# Find the header row by looking for required columns
		header_row_index = -1
		for i, line in enumerate(lines):
			# Try to parse this line as potential header
			try:
				# Use csv.reader to properly handle quoted fields
				row_reader = csv.reader([line])
				row = next(row_reader)
				# Check if this row contains the required columns
				if (symbol_column in row and
				    quantity_column in row and
				    total_cost_column in row and
				    any(col.startswith(market_value_column_prefix) for col in row)):
					header_row_index = i
					break
			except:
				continue

		if header_row_index == -1:
			raise Exception(f'Could not find header row with required columns: {symbol_column}, {quantity_column}, {total_cost_column}, and {market_value_column_prefix}*')

		# Reset file and skip to header row
		file.seek(0)
		for _ in range(header_row_index):
			next(file)

		# Now read with DictReader starting from the header
		reader = csv.DictReader(file)

		# Find the market value column name
		market_value_column = ''
		for header in reader.fieldnames:
			if header.startswith(market_value_column_prefix):
				market_value_column = header
				break

		if not market_value_column:
			raise Exception(f'Could not find a column that starts with prefix "{market_value_column_prefix}"')

		# Parse the data rows
		for row in reader:
			# Skip empty rows or rows with missing data
			if not row.get(symbol_column) or not row.get(quantity_column):
				continue

			try:
				ticker = row[symbol_column].strip()
				quantity = float(row[quantity_column].replace(',', ''))
				cost_basis = float(row[total_cost_column].replace(',', '').replace('$', ''))
				market_value = float(row[market_value_column].replace(',', '').replace('$', ''))

				if ticker not in lots_by_ticker:
					lots_by_ticker[ticker] = []
				lots_by_ticker[ticker].append(Lot(quantity, cost_basis, market_value))
			except (ValueError, KeyError) as e:
				# Skip rows that can't be parsed (might be subtotals or other non-data rows)
				continue

		# Sort lots by cost basis per share (MinTax strategy)
		for lots in lots_by_ticker.values():
			lots.sort(key=lambda x: x.cost_basis/x.quantity)

	return lots_by_ticker

def parse_cost_basis_vanguard_csv_by_ticker(csv_file_path):

	lots_by_ticker = parse_cost_basis_vanguard_csv(csv_file_path)

	return [{'ticker': ticker, 'quantity': (sum(lot.quantity for lot in lots)), 'market_value': (sum(lot.market_value for lot in lots))} for ticker, lots in lots_by_ticker.items()]
