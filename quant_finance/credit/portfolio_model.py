#### Credit Portfolio Model ####

from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant_finance.credit.portfolio_input import (
    load_portfolio,
    load_model_settings,
    load_factor_proxies,
    load_rating_migration_matrix,
    load_market_data,
    validate_portfolio,
    validate_model_settings,
    validate_factor_proxies,
    validate_rating_migration_matrix,
    validate_market_data,
)

from quant_finance.credit.counterparty_rating import get_counterparty_pd
from quant_finance.credit.merton import merton_probability_of_default
from quant_finance.credit.merton_calibration import calibrate_merton
from quant_finance.credit.merton_market_data import (
    estimate_equity_value,
    estimate_equity_volatility,
    estimate_debt_default_point,
)
from quant_finance.credit.factor_market_data import estimate_counterparty_factor_loadings
from quant_finance.credit.portfolio_simulation import simulate_migration_portfolio_losses
from quant_finance.credit.credit_risk_analytics import (
    expected_loss,
    value_at_risk,
    expected_shortfall,
    unexpected_loss,
    incremental_risk,
    marginal_risk,
    diversification_benefit,
)

from quant_finance.market_data.rates import get_risk_free_rate
from quant_finance.market_data.fx import get_fx_rate


# Merton PD horizon.
MERTON_HORIZON = 1.0


def _has_text(value):
    return pd.notna(value) and str(value).strip() != ""


# Retrieve risk-free rates automatically.
# Use the frozen Excel snapshot only if automatic retrieval fails.
def _complete_risk_free_rates(portfolio, market_data):

    portfolio["risk_free_rate"] = np.nan
    portfolio["risk_free_rate_source"] = None

    rate_cache = {}

    for index, row in portfolio.iterrows():

        currency = str(row["currency"]).upper().strip()

        if currency not in rate_cache:
            rate_cache[currency] = get_risk_free_rate(
                currency=currency,
                market_data=market_data,
                return_source=True,
            )

        rate, source = rate_cache[currency]

        portfolio.at[index, "risk_free_rate"] = rate
        portfolio.at[index, "risk_free_rate_source"] = source

    return portfolio


# Convert portfolio exposures into the model base currency.
def _convert_exposures_to_base_currency(
    portfolio,
    base_currency,
    market_data,
):

    base_currency = str(base_currency).upper().strip()

    portfolio["exposure_local"] = portfolio["exposure"].astype(float)
    portfolio["fx_rate_to_base"] = np.nan
    portfolio["fx_rate_source"] = None
    portfolio["exposure_base"] = np.nan

    fx_cache = {}

    for index, row in portfolio.iterrows():

        currency = str(row["currency"]).upper().strip()

        if currency not in fx_cache:
            fx_cache[currency] = get_fx_rate(
                currency=currency,
                base_currency=base_currency,
                market_data=market_data,
                return_source=True,
            )

        fx_rate, source = fx_cache[currency]

        exposure_base = float(row["exposure"]) * fx_rate

        portfolio.at[index, "fx_rate_to_base"] = fx_rate
        portfolio.at[index, "fx_rate_source"] = source
        portfolio.at[index, "exposure_base"] = exposure_base

    # Downstream credit functions use exposure in base currency.
    portfolio["exposure"] = portfolio["exposure_base"]

    return portfolio


# Calculate rating-based or Merton-based PD for each counterparty.
def _calculate_counterparty_pds(
    portfolio,
    start_date,
    end_date,
):

    portfolio["pd"] = np.nan
    portfolio["pd_source"] = None
    portfolio["asset_value"] = np.nan
    portfolio["asset_volatility"] = np.nan

    for index, row in portfolio.iterrows():

        method = str(row["pd_method"]).lower()

        # Rating-based PD.
        if method == "rating":

            portfolio.at[index, "pd"] = get_counterparty_pd(
                method="rating",
                rating=row["rating"],
            )

            portfolio.at[index, "pd_source"] = "rating"

            continue

        if method != "merton":
            raise ValueError(
                f"Unsupported PD method '{row['pd_method']}' "
                f"for counterparty '{row['counterparty']}'."
            )

        ticker = (
            str(row["ticker"]).strip()
            if _has_text(row["ticker"])
            else None
        )

        equity_value = row["equity_value"]
        equity_volatility = row["equity_volatility"]
        debt = row["debt"]

        manual_equity_value = pd.notna(equity_value)
        manual_equity_volatility = pd.notna(equity_volatility)
        manual_debt = pd.notna(debt)

        manual_complete = (
            manual_equity_value
            and manual_equity_volatility
            and manual_debt
        )

        # Fully manual company inputs.
        if manual_complete:

            pd_source = "merton_manual"

        # Otherwise obtain missing company inputs from market data.
        elif ticker is not None:

            if not manual_equity_value:

                equity_value = estimate_equity_value(ticker)

                portfolio.at[
                    index,
                    "equity_value",
                ] = equity_value

            if not manual_equity_volatility:

                equity_volatility = estimate_equity_volatility(
                    ticker,
                    start_date,
                    end_date,
                )

                portfolio.at[
                    index,
                    "equity_volatility",
                ] = equity_volatility

            if not manual_debt:

                debt, _, _ = estimate_debt_default_point(ticker)

                portfolio.at[
                    index,
                    "debt",
                ] = debt

            if (
                manual_equity_value
                or manual_equity_volatility
                or manual_debt
            ):
                pd_source = "merton_mixed"

            else:
                pd_source = "merton_market_data"

        else:
            raise ValueError(
                f"Counterparty '{row['counterparty']}' requires either "
                f"complete manual Merton inputs or a ticker."
            )

        # Merton company values remain in the company's original
        # currency. Portfolio exposure is converted separately.
        risk_free_rate = float(
            portfolio.at[
                index,
                "risk_free_rate",
            ]
        )

        asset_value, asset_volatility = calibrate_merton(
            equity_value=float(equity_value),
            equity_volatility=float(equity_volatility),
            debt=float(debt),
            risk_free_rate=risk_free_rate,
            maturity=MERTON_HORIZON,
        )

        merton_pd = merton_probability_of_default(
            asset_value=asset_value,
            debt_face_value=float(debt),
            risk_free_rate=risk_free_rate,
            asset_volatility=asset_volatility,
            time_to_maturity=MERTON_HORIZON,
        )

        portfolio.at[
            index,
            "asset_value",
        ] = asset_value

        portfolio.at[
            index,
            "asset_volatility",
        ] = asset_volatility

        portfolio.at[
            index,
            "pd",
        ] = get_counterparty_pd(
            method="merton",
            merton_pd=merton_pd,
        )

        portfolio.at[
            index,
            "pd_source",
        ] = pd_source

    return portfolio


# Calculate expected loss for each counterparty.
def _calculate_expected_losses(portfolio):

    portfolio["expected_loss"] = [
        expected_loss(
            exposure,
            pd_value,
            lgd,
        )
        for exposure, pd_value, lgd in zip(
            portfolio["exposure"],
            portfolio["pd"],
            portfolio["lgd"],
        )
    ]

    return portfolio


# Convert factor-model output into the four expected loadings.
def _unpack_factor_loadings(loadings):

    if isinstance(loadings, dict):

        return (
            loadings["global_loading"],
            loadings["region_loading"],
            loadings["sector_loading"],
            loadings["idiosyncratic_loading"],
        )

    if len(loadings) != 4:
        raise ValueError(
            "Factor loading estimation must return four loadings."
        )

    return loadings


# Estimate systematic factor loadings for every counterparty.
#
# Listed counterparties use their own ticker first.
# If factor-market-data estimation fails, an explicitly supplied
# proxy ticker is used as fallback.
#
# Unlisted counterparties use their proxy ticker directly.
def _estimate_portfolio_factor_loadings(
    portfolio,
    factor_proxies,
    start_date,
    end_date,
):

    portfolio["factor_calibration_ticker"] = None
    portfolio["factor_source"] = None
    portfolio["global_loading"] = np.nan
    portfolio["region_loading"] = np.nan
    portfolio["sector_loading"] = np.nan
    portfolio["idiosyncratic_loading"] = np.nan

    for index, row in portfolio.iterrows():

        ticker = (
            str(row["ticker"]).strip()
            if _has_text(row["ticker"])
            else None
        )

        proxy_ticker = (
            str(row["factor_loading_proxy_ticker"]).strip()
            if _has_text(row["factor_loading_proxy_ticker"])
            else None
        )

        # Listed counterparty: try own ticker first.
        if ticker is not None:

            try:

                loadings = estimate_counterparty_factor_loadings(
                    company_ticker=ticker,
                    region=row["region"],
                    sector=row["sector"],
                    factor_proxies=factor_proxies,
                    start_date=start_date,
                    end_date=end_date,
                )

                calibration_ticker = ticker
                factor_source = "own_ticker"

            except ValueError as own_ticker_error:

                if proxy_ticker is None:
                    raise ValueError(
                        f"Factor loading estimation failed for "
                        f"counterparty '{row['counterparty']}' using "
                        f"ticker '{ticker}', and no fallback proxy ticker "
                        f"is available. Original error: "
                        f"{own_ticker_error}"
                    ) from own_ticker_error

                loadings = estimate_counterparty_factor_loadings(
                    company_ticker=proxy_ticker,
                    region=row["region"],
                    sector=row["sector"],
                    factor_proxies=factor_proxies,
                    start_date=start_date,
                    end_date=end_date,
                )

                calibration_ticker = proxy_ticker
                factor_source = "fallback_proxy_ticker"

        # Unlisted counterparty: use proxy directly.
        elif proxy_ticker is not None:

            loadings = estimate_counterparty_factor_loadings(
                company_ticker=proxy_ticker,
                region=row["region"],
                sector=row["sector"],
                factor_proxies=factor_proxies,
                start_date=start_date,
                end_date=end_date,
            )

            calibration_ticker = proxy_ticker
            factor_source = "proxy_ticker"

        else:
            raise ValueError(
                f"No factor calibration ticker available for "
                f"counterparty '{row['counterparty']}'."
            )

        (
            global_loading,
            region_loading,
            sector_loading,
            idiosyncratic_loading,
        ) = _unpack_factor_loadings(loadings)

        portfolio.at[
            index,
            "factor_calibration_ticker",
        ] = calibration_ticker

        portfolio.at[
            index,
            "factor_source",
        ] = factor_source

        portfolio.at[
            index,
            "global_loading",
        ] = global_loading

        portfolio.at[
            index,
            "region_loading",
        ] = region_loading

        portfolio.at[
            index,
            "sector_loading",
        ] = sector_loading

        portfolio.at[
            index,
            "idiosyncratic_loading",
        ] = idiosyncratic_loading

    return portfolio


# Calculate leave-one-out counterparty risk contributions
# using the same Monte Carlo scenarios.
def _calculate_risk_contributions(
    portfolio,
    portfolio_losses,
    counterparty_losses,
    confidence_level,
    portfolio_var,
    portfolio_es,
):

    result_columns = [
        "incremental_var",
        "incremental_es",
        "marginal_var",
        "marginal_es",
        "standalone_var",
        "standalone_es",
        "var_diversification_benefit",
        "es_diversification_benefit",
    ]

    for column in result_columns:
        portfolio[column] = np.nan

    for column_index, (index, row) in enumerate(
        portfolio.iterrows()
    ):

        counterparty_loss = counterparty_losses[
            :,
            column_index,
        ]

        portfolio_without = (
            portfolio_losses
            - counterparty_loss
        )

        var_without = value_at_risk(
            portfolio_without,
            confidence_level,
        )

        es_without = expected_shortfall(
            portfolio_without,
            confidence_level,
        )

        incremental_var = incremental_risk(
            var_without,
            portfolio_var,
        )

        incremental_es = incremental_risk(
            es_without,
            portfolio_es,
        )

        marginal_var = marginal_risk(
            var_without,
            portfolio_var,
            float(row["exposure"]),
        )

        marginal_es = marginal_risk(
            es_without,
            portfolio_es,
            float(row["exposure"]),
        )

        standalone_var = value_at_risk(
            counterparty_loss,
            confidence_level,
        )

        standalone_es = expected_shortfall(
            counterparty_loss,
            confidence_level,
        )

        portfolio.at[
            index,
            "incremental_var",
        ] = incremental_var

        portfolio.at[
            index,
            "incremental_es",
        ] = incremental_es

        portfolio.at[
            index,
            "marginal_var",
        ] = marginal_var

        portfolio.at[
            index,
            "marginal_es",
        ] = marginal_es

        portfolio.at[
            index,
            "standalone_var",
        ] = standalone_var

        portfolio.at[
            index,
            "standalone_es",
        ] = standalone_es

        portfolio.at[
            index,
            "var_diversification_benefit",
        ] = diversification_benefit(
            standalone_var,
            incremental_var,
        )

        portfolio.at[
            index,
            "es_diversification_benefit",
        ] = diversification_benefit(
            standalone_es,
            incremental_es,
        )

    return portfolio


# Run the complete credit portfolio model.
def run_credit_portfolio_model(file_path):

    # Load Excel steering inputs.
    portfolio = load_portfolio(file_path)
    settings = load_model_settings(file_path)
    factor_proxies = load_factor_proxies(file_path)
    rating_migration_matrix = load_rating_migration_matrix(
        file_path
    )
    market_data = load_market_data(file_path)

    # Validate raw model inputs.
    validate_portfolio(portfolio)
    validate_model_settings(settings)
    validate_factor_proxies(factor_proxies)
    validate_rating_migration_matrix(
        rating_migration_matrix
    )

    base_currency = str(
        settings["base_currency"]
    ).upper().strip()

    validate_market_data(
        market_data,
        base_currency=base_currency,
    )

    # Dates used for market-data estimation.
    end_date = date.today()

    factor_lookback_years = int(
        settings["factor_lookback_years"]
    )

    factor_start_date = (
        end_date
        - timedelta(
            days=365 * factor_lookback_years
        )
    )

    # One-year history for Merton equity volatility.
    merton_start_date = (
        end_date
        - timedelta(days=365)
    )

    # Risk-free rates:
    # automatic market data first,
    # frozen snapshot only as fallback.
    portfolio = _complete_risk_free_rates(
        portfolio,
        market_data=market_data,
    )

    # FX:
    # automatic ECB data first,
    # frozen snapshot only as fallback.
    portfolio = _convert_exposures_to_base_currency(
        portfolio,
        base_currency=base_currency,
        market_data=market_data,
    )

    # Counterparty PDs and expected losses.
    portfolio = _calculate_counterparty_pds(
        portfolio,
        start_date=merton_start_date,
        end_date=end_date,
    )

    portfolio = _calculate_expected_losses(
        portfolio
    )

    # Estimate dependence-model factor loadings.
    portfolio = _estimate_portfolio_factor_loadings(
        portfolio=portfolio,
        factor_proxies=factor_proxies,
        start_date=factor_start_date,
        end_date=end_date,
    )

    # Simulate correlated migration/default losses.
    (
        portfolio_losses,
        migration_states,
        counterparty_losses,
    ) = simulate_migration_portfolio_losses(
        portfolio=portfolio,
        settings=settings,
        rating_migration_matrix=rating_migration_matrix,
        return_states=True,
        return_counterparty_losses=True,
    )

    confidence_level = float(
        settings["confidence_level"]
    )

    # Portfolio risk metrics.
    default_expected_loss = float(
        portfolio["expected_loss"].sum()
    )

    migration_mean_loss = float(
        np.mean(portfolio_losses)
    )

    portfolio_var = value_at_risk(
        portfolio_losses,
        confidence_level,
    )

    portfolio_es = expected_shortfall(
        portfolio_losses,
        confidence_level,
    )

    portfolio_ul = unexpected_loss(
        portfolio_var,
        migration_mean_loss,
    )

    # Counterparty risk contributions.
    portfolio = _calculate_risk_contributions(
        portfolio=portfolio,
        portfolio_losses=portfolio_losses,
        counterparty_losses=counterparty_losses,
        confidence_level=confidence_level,
        portfolio_var=portfolio_var,
        portfolio_es=portfolio_es,
    )

    summary = {
        "base_currency": base_currency,
        "default_expected_loss": default_expected_loss,
        "migration_mean_loss": migration_mean_loss,
        "confidence_level": confidence_level,
        "value_at_risk": portfolio_var,
        "expected_shortfall": portfolio_es,
        "unexpected_loss": portfolio_ul,
        "maximum_simulated_loss": float(
            np.max(portfolio_losses)
        ),
        "minimum_simulated_loss": float(
            np.min(portfolio_losses)
        ),
    }

    return {
        "portfolio": portfolio,
        "settings": settings,
        "factor_proxies": factor_proxies,
        "rating_migration_matrix": rating_migration_matrix,
        "market_data": market_data,
        "portfolio_losses": portfolio_losses,
        "migration_states": migration_states,
        "counterparty_losses": counterparty_losses,
        "summary": summary,
    }