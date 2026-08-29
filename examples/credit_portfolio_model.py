#### Credit Portfolio Model Example ####

from datetime import date, timedelta

import pandas as pd

from quant_finance.credit.portfolio_input import (
    load_portfolio,
    validate_portfolio,
)

from quant_finance.credit.counterparty_rating import (
    get_counterparty_pd,
)

from quant_finance.credit.merton_calibration import (
    calibrate_merton,
)

from quant_finance.credit.merton import (
    merton_probability_of_default,
)

from quant_finance.credit.merton_market_data import (
    estimate_equity_volatility,
    estimate_equity_value,
    estimate_debt_default_point,
)

from quant_finance.market_data.rates import (
    get_risk_free_rate,
)


# Merton PD horizon in years.

MERTON_HORIZON = 1.0


# Historical lookback period used to estimate equity volatility.

MARKET_DATA_END_DATE = date.today()

MARKET_DATA_START_DATE = (
    MARKET_DATA_END_DATE
    - timedelta(days=365)
)


# Load and validate portfolio input.

portfolio = load_portfolio(
    "data/credit_risk_input.xlsx"
)

validate_portfolio(portfolio)


# Create result columns.

portfolio["pd"] = float("nan")
portfolio["pd_source"] = None

portfolio["asset_value"] = float("nan")
portfolio["asset_volatility"] = float("nan")


# Track the source of the Merton input parameters.

portfolio["equity_value_source"] = None
portfolio["equity_volatility_source"] = None
portfolio["debt_source"] = None
portfolio["risk_free_rate_source"] = None


# Calculate counterparty PDs.

for index, row in portfolio.iterrows():

    # Rating-based PD.
    if row["pd_method"] == "rating":

        pd_value = get_counterparty_pd(
            method="rating",
            rating=row["rating"],
        )

        portfolio.at[index, "pd"] = pd_value
        portfolio.at[index, "pd_source"] = "rating"


    # Merton-based PD.
    elif row["pd_method"] == "merton":

        ticker = None

        if (
            pd.notna(row["ticker"])
            and str(row["ticker"]).strip() != ""
        ):
            ticker = str(
                row["ticker"]
            ).strip()


        # Equity value.

        if pd.notna(
            row["equity_value"]
        ):

            equity_value = row[
                "equity_value"
            ]

            equity_value_source = "manual"

        else:

            equity_value = estimate_equity_value(
                ticker=ticker
            )

            equity_value_source = "market_data"


        # Equity volatility.

        if pd.notna(
            row["equity_volatility"]
        ):

            equity_volatility = row[
                "equity_volatility"
            ]

            equity_volatility_source = "manual"

        else:

            equity_volatility = (
                estimate_equity_volatility(
                    ticker=ticker,
                    start_date=MARKET_DATA_START_DATE,
                    end_date=MARKET_DATA_END_DATE,
                )
            )

            equity_volatility_source = (
                "market_data"
            )


        # Merton debt default point.

        if pd.notna(
            row["debt"]
        ):

            debt = row[
                "debt"
            ]

            debt_source = "manual"

        else:

            (
                debt,
                short_term_debt,
                long_term_debt,
            ) = estimate_debt_default_point(
                ticker=ticker
            )

            debt_source = "market_data"


        # Risk-free rate.

        if pd.notna(
            row["risk_free_rate"]
        ):

            risk_free_rate = row[
                "risk_free_rate"
            ]

            risk_free_rate_source = "manual"

        else:

            risk_free_rate = get_risk_free_rate(
                currency=row["currency"]
            )

            risk_free_rate_source = (
                "market_data"
            )


        # Calibrate Merton asset value and asset volatility.

        asset_value, asset_volatility = (
            calibrate_merton(
                equity_value=equity_value,
                equity_volatility=equity_volatility,
                debt=debt,
                risk_free_rate=risk_free_rate,
                maturity=MERTON_HORIZON,
            )
        )


        # Calculate one-year Merton probability of default.

        pd_value = (
            merton_probability_of_default(
                asset_value=asset_value,
                debt_face_value=debt,
                risk_free_rate=risk_free_rate,
                asset_volatility=asset_volatility,
                time_to_maturity=MERTON_HORIZON,
            )
        )


        # Store Merton input values.

        portfolio.at[
            index,
            "equity_value",
        ] = equity_value

        portfolio.at[
            index,
            "equity_volatility",
        ] = equity_volatility

        portfolio.at[
            index,
            "debt",
        ] = debt

        portfolio.at[
            index,
            "risk_free_rate",
        ] = risk_free_rate


        # Store calibrated Merton values.

        portfolio.at[
            index,
            "asset_value",
        ] = asset_value

        portfolio.at[
            index,
            "asset_volatility",
        ] = asset_volatility


        # Store parameter sources.

        portfolio.at[
            index,
            "equity_value_source",
        ] = equity_value_source

        portfolio.at[
            index,
            "equity_volatility_source",
        ] = equity_volatility_source

        portfolio.at[
            index,
            "debt_source",
        ] = debt_source

        portfolio.at[
            index,
            "risk_free_rate_source",
        ] = risk_free_rate_source


        # Determine overall PD source.

        input_sources = {
            equity_value_source,
            equity_volatility_source,
            debt_source,
            risk_free_rate_source,
        }

        if input_sources == {"manual"}:

            pd_source = "merton_manual"

        elif input_sources == {"market_data"}:

            pd_source = "merton_market_data"

        else:

            pd_source = "merton_mixed"


        portfolio.at[
            index,
            "pd",
        ] = pd_value

        portfolio.at[
            index,
            "pd_source",
        ] = pd_source


# Calculate expected loss for each counterparty.

portfolio["expected_loss"] = (
    portfolio["exposure"]
    * portfolio["pd"]
    * portfolio["lgd"]
)


# Calculate total portfolio expected loss.

portfolio_expected_loss = (
    portfolio["expected_loss"].sum()
)


# Display current portfolio results.

print(
    "\nCredit portfolio results:"
)

print(
    portfolio[
        [
            "counterparty",
            "ticker",
            "pd_method",
            "exposure",
            "lgd",
            "pd",
            "pd_source",
            "expected_loss",
            "equity_value",
            "equity_volatility",
            "debt",
            "risk_free_rate",
            "asset_value",
            "asset_volatility",
        ]
    ].to_string(
        index=False
    )
)


# Display Merton input sources.

print(
    "\nMerton input sources:"
)

print(
    portfolio[
        [
            "counterparty",
            "equity_value_source",
            "equity_volatility_source",
            "debt_source",
            "risk_free_rate_source",
        ]
    ].to_string(
        index=False
    )
)


# Display portfolio expected loss.

print(
    f"\nPortfolio expected loss: "
    f"{portfolio_expected_loss:,.2f}"
)