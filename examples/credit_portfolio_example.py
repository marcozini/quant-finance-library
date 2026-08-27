#### Credit Portfolio Input Example ####

from quant_finance.credit.portfolio_input import (
    load_portfolio,
    validate_portfolio,
)

# Load portfolio from Excel.

portfolio = load_portfolio(
    "data/credit_risk_input.xlsx"
)

validate_portfolio(portfolio)

print("\nPortfolio validation passed.")

# Display portfolio data.

print(portfolio)


# Display data types for a first input check.

print("\nData types:")
print(portfolio.dtypes)