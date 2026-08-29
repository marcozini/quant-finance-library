#### Merton Market Data Example ####

from quant_finance.credit.merton_market_data import (
    estimate_equity_volatility,
    estimate_equity_value,
    estimate_debt_default_point,
)


# Estimate equity volatility from historical market data.

equity_volatility = estimate_equity_volatility(
    ticker="AAPL",
    start_date="2025-01-01",
    end_date="2026-01-01",
)


# Estimate market value of equity.

equity_value = estimate_equity_value(
    ticker="AAPL"
)


# Estimate Merton debt default point from balance-sheet data.

debt, short_term_debt, long_term_debt = (
    estimate_debt_default_point(
        ticker="AAPL"
    )
)


# Display results.

print(
    f"Estimated annualized equity volatility: "
    f"{equity_volatility:.2%}"
)

print(
    f"Estimated equity value: "
    f"{equity_value:,.0f}"
)

print(
    f"Short-term debt: "
    f"{short_term_debt:,.0f}"
)

print(
    f"Long-term debt: "
    f"{long_term_debt:,.0f}"
)

print(
    f"Merton debt default point: "
    f"{debt:,.0f}"
)