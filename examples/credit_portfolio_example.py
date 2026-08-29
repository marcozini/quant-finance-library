#### Credit Portfolio Input Example ####

from quant_finance.credit.portfolio_input import (
    load_portfolio,
    load_model_settings,
    validate_portfolio,
    validate_model_settings,
    load_factor_proxies,
    validate_factor_proxies,
)


# Load and validate portfolio.

portfolio = load_portfolio("data/credit_risk_input.xlsx")
validate_portfolio(portfolio)

print("\nPortfolio validation passed.")


# Load and validate model settings.

settings = load_model_settings("data/credit_risk_input.xlsx")
validate_model_settings(settings)

print("\nModel settings validation passed.")


# Display portfolio data.

print("\nPortfolio:")
print(portfolio)


# Display portfolio data types.

print("\nPortfolio data types:")
print(portfolio.dtypes)


# Display model settings.

print("\nModel settings:")

for setting, value in settings.items():
    print(f"{setting}: {value}")
    
    
# Load and validate factor proxies.

factor_proxies = load_factor_proxies(
    "data/credit_risk_input.xlsx"
)

validate_factor_proxies(factor_proxies)

print("\nFactor proxy validation passed.")

print("\nFactor proxies:")
print(factor_proxies)