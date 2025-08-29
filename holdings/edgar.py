from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import os

# Vanguard 500 Index Fund CIK
CIK = "0000038275"

dl = Downloader(email_address="my_email@example.com", company_name=CIK)

# Download the latest N-PORT filing
dl.get(form="NPORT-P", ticker_or_cik=CIK)

# Locate filing
base_path = f"./sec-edgar-filings/NPORT-P/{CIK}/"
latest_filing = sorted(os.listdir(base_path))[-1]
filing_path = os.path.join(base_path, latest_filing, "primary-document.xml")

# Parse holdings
with open(filing_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "xml")

holdings = []
for inv in soup.find_all("invstOrSec"):
    holdings.append({
        "name": inv.find("name").get_text(strip=True) if inv.find("name") else None,
        "cusip": inv.find("cusip").get_text(strip=True) if inv.find("cusip") else None,
        "shares": inv.find("balance").get_text(strip=True) if inv.find("balance") else None,
        "value_usd": inv.find("valueUSD").get_text(strip=True) if inv.find("valueUSD") else None,
    })

print(f"Found {len(holdings)} holdings")
print(holdings[:5])