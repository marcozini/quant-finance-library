#### Credit Portfolio Model Example ####

from quant_finance.credit.portfolio_input import (
    load_portfolio,
    validate_portfolio,
)

from quant_finance.credit.counterparty_rating import (
    get_counterparty_pd,
)


# Load and validate portfolio input.

portfolio = load_portfolio(
    "data/credit_risk_input.xlsx"
)

validate_portfolio(portfolio)


# Create result columns.

portfolio["pd"] = float("nan")
portfolio["pd_source"] = None


# Calculate PD for rating-based counterparties.

for index, row in portfolio.iterrows():

    if row["pd_method"] == "rating":

        pd_value = get_counterparty_pd(
            method="rating",
            rating=row["rating"],
        )

        portfolio.at[index, "pd"] = pd_value
        portfolio.at[index, "pd_source"] = "rating"


# Display current portfolio results.

print("\nCredit portfolio results:")
print(
    portfolio[
        [
            "counterparty",
            "pd_method",
            "rating",
            "pd",
            "pd_source",
        ]
    ]
)