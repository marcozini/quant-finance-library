#### Risk-Free Rate Example ####

from quant_finance.market_data.rates import (
    get_risk_free_rate,
)


currencies = [
    "USD",
    "EUR",
    "GBP",
    "CHF",
]


# Retrieve and display risk-free rates.

for currency in currencies:

    try:

        risk_free_rate = get_risk_free_rate(
            currency=currency
        )

        print(
            f"{currency} risk-free rate: "
            f"{risk_free_rate:.2%}"
        )

    except Exception as error:

        print(
            f"{currency}: ERROR - {error}"
        )