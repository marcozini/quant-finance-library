#### Unit Tests Credit Portfolio Input ####

from datetime import date, timedelta

import pandas as pd
import pytest

from quant_finance.credit.portfolio_input import (
    parse_market_data_date,
    validate_portfolio,
    validate_model_settings,
    validate_factor_proxies,
    validate_market_data,
    validate_rating_migration_matrix,
)


# =========================================================
# Test Data Helpers
# =========================================================

def create_valid_portfolio():

    return pd.DataFrame({

        "counterparty": [
            "Listed Co",
            "Private Co",
        ],

        "ticker": [
            "TEST",
            None,
        ],

        "factor_loading_proxy_ticker": [
            None,
            "PROXY",
        ],

        "exposure": [
            100.0,
            50.0,
        ],

        "currency": [
            "USD",
            "EUR",
        ],

        "lgd": [
            0.50,
            0.50,
        ],

        "pd_method": [
            "rating",
            "merton",
        ],

        "rating": [
            "AA",
            None,
        ],

        "equity_value": [
            None,
            20.0,
        ],

        "equity_volatility": [
            None,
            0.45,
        ],

        "debt": [
            None,
            80.0,
        ],

        "maturity": [
            5.0,
            4.0,
        ],

        "coupon_rate": [
            0.04,
            0.05,
        ],

        "payment_frequency": [
            2,
            1,
        ],

        "sector": [
            "Technology",
            "Industrial",
        ],

        "region": [
            "US",
            "Europe",
        ],
    })


def create_valid_settings():

    return {

        "number_simulations": 100000,

        "seed": 0,

        "confidence_level": 0.995,

        "dependence_model": "t_copula",

        "t_degrees_of_freedom": 5,

        "factor_structure": "global_sector_region",

        "factor_lookback_years": 3,

        "base_currency": "USD",

        # Use latest available market inputs by default.
        "market_data_date": "latest",

        "create_excel_output": True,

        "create_pdf_output": True,

        "create_plot_files": True,

        "output_folder": "outputs/",
    }


def create_valid_factor_proxies():

    return pd.DataFrame({

        "factor_type": [
            "global",
            "region",
            "region",
            "sector",
        ],

        "factor_name": [
            "Global",
            "US",
            "Europe",
            "Technology",
        ],

        "ticker": [
            "ACWI",
            "SPY",
            "VGK",
            "IXN",
        ],

        "description": [
            "Global market proxy.",
            "US regional proxy.",
            "European regional proxy.",
            "Technology sector proxy.",
        ],
    })


def create_valid_market_data():

    return pd.DataFrame({

        "data_type": [
            "fx",
            "fx",
            "fx",
            "fx",
            "risk_free",
            "risk_free",
            "risk_free",
            "risk_free",
        ],

        "currency": [
            "CHF",
            "USD",
            "EUR",
            "GBP",
            "CHF",
            "USD",
            "EUR",
            "GBP",
        ],

        "base_currency": [
            "CHF",
            "CHF",
            "CHF",
            "CHF",
            None,
            None,
            None,
            None,
        ],

        "tenor_years": [
            None,
            None,
            None,
            None,
            1.0,
            1.0,
            1.0,
            1.0,
        ],

        "value": [
            1.0,
            0.80,
            0.94,
            1.07,
            0.00,
            0.04,
            0.02,
            0.04,
        ],

        "as_of_date": [
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
        ],

        "curve_type": [
            None,
            None,
            None,
            None,
            "flat_policy_proxy",
            "flat_policy_proxy",
            "flat_policy_proxy",
            "flat_policy_proxy",
        ],

        "source": [
            "Demo snapshot",
            "Demo snapshot",
            "Demo snapshot",
            "Demo snapshot",
            "Demo fallback",
            "Demo fallback",
            "Demo fallback",
            "Demo fallback",
        ],
    })


def create_valid_rating_migration_matrix():

    ratings = [
        "AAA",
        "AA",
        "A",
        "BBB",
        "BB",
        "B",
        "CCC",
        "D",
    ]

    transition_matrix = [

        [
            0.90,
            0.08,
            0.01,
            0.005,
            0.003,
            0.001,
            0.0005,
            0.0005,
        ],

        [
            0.02,
            0.90,
            0.06,
            0.01,
            0.005,
            0.002,
            0.001,
            0.002,
        ],

        [
            0.005,
            0.03,
            0.88,
            0.06,
            0.015,
            0.005,
            0.002,
            0.003,
        ],

        [
            0.002,
            0.008,
            0.04,
            0.84,
            0.07,
            0.025,
            0.008,
            0.007,
        ],

        [
            0.001,
            0.003,
            0.01,
            0.05,
            0.80,
            0.09,
            0.025,
            0.021,
        ],

        [
            0.001,
            0.001,
            0.003,
            0.01,
            0.06,
            0.76,
            0.09,
            0.075,
        ],

        [
            0.0005,
            0.0005,
            0.001,
            0.003,
            0.01,
            0.05,
            0.70,
            0.235,
        ],

        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ]

    spreads_bps = [
        20,
        40,
        70,
        120,
        250,
        500,
        1000,
        None,
    ]

    data = {

        "current_rating": ratings,

        "spread_bps": spreads_bps,

        "spread_decimal": [
            0.002,
            0.004,
            0.007,
            0.012,
            0.025,
            0.050,
            0.100,
            None,
        ],
    }

    for column_index, rating in enumerate(ratings):

        data[rating] = [
            row[column_index]
            for row in transition_matrix
        ]

    data["row_sum"] = [
        sum(row)
        for row in transition_matrix
    ]

    return pd.DataFrame(data)


# =========================================================
# Portfolio Validation
# =========================================================

# 1. Valid portfolio passes.
def test_valid_portfolio():

    assert validate_portfolio(
        create_valid_portfolio()
    )


# 2. risk_free_rate is no longer required in the portfolio input.
def test_portfolio_does_not_require_risk_free_rate():

    portfolio = create_valid_portfolio()

    assert "risk_free_rate" not in portfolio.columns

    assert validate_portfolio(
        portfolio
    )


# 3. Missing required portfolio column raises an error.
def test_missing_portfolio_column():

    portfolio = (
        create_valid_portfolio()
        .drop(columns=["exposure"])
    )

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# 4. Exposure must be positive.
def test_invalid_exposure():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "exposure",
    ] = 0

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# 5. LGD must lie between zero and one.
def test_invalid_lgd():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "lgd",
    ] = 1.20

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# 6. Currency is required.
def test_missing_currency():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "currency",
    ] = None

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# 7. PD method must be valid.
def test_invalid_pd_method():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "pd_method",
    ] = "invalid"

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# 8. Rating-based counterparties require a rating.
def test_rating_method_requires_rating():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "rating",
    ] = None

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# 9. Manual Merton inputs work without a risk-free-rate input.
def test_manual_merton_inputs_without_risk_free_rate():

    portfolio = create_valid_portfolio()

    assert validate_portfolio(
        portfolio
    )


# 10. Incomplete Merton inputs without ticker raise an error.
def test_incomplete_manual_merton_inputs():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        1,
        "equity_value",
    ] = None

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# 11. Unlisted counterparty requires a factor proxy ticker.
def test_unlisted_counterparty_requires_factor_proxy():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        1,
        "factor_loading_proxy_ticker",
    ] = None

    with pytest.raises(ValueError):

        validate_portfolio(
            portfolio
        )


# =========================================================
# Model Settings Validation
# =========================================================

# 12. Valid model settings pass.
def test_valid_model_settings():

    assert validate_model_settings(
        create_valid_settings()
    )


# 13. Invalid base currency raises an error.
def test_invalid_base_currency():

    settings = create_valid_settings()

    settings[
        "base_currency"
    ] = "US"

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Valid dependence model choices.
@pytest.mark.parametrize(
    "dependence_model",
    [
        "independent",
        "gaussian_copula",
        "t_copula",
    ],
)
def test_valid_dependence_models(
    dependence_model,
):

    settings = create_valid_settings()

    settings[
        "dependence_model"
    ] = dependence_model

    assert validate_model_settings(
        settings
    )


# Invalid dependence model raises an error.
def test_invalid_dependence_model():

    settings = create_valid_settings()

    settings[
        "dependence_model"
    ] = "invalid"

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Valid factor structure choices.
@pytest.mark.parametrize(
    "factor_structure",
    [
        "single_factor",
        "global_sector",
        "global_region",
        "global_sector_region",
    ],
)
def test_valid_factor_structures(
    factor_structure,
):

    settings = create_valid_settings()

    settings[
        "factor_structure"
    ] = factor_structure

    assert validate_model_settings(
        settings
    )


# Invalid factor structure raises an error.
def test_invalid_factor_structure():

    settings = create_valid_settings()

    settings[
        "factor_structure"
    ] = "invalid"

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Every valid dependence and factor combination should pass.
@pytest.mark.parametrize(
    "dependence_model",
    [
        "independent",
        "gaussian_copula",
        "t_copula",
    ],
)
@pytest.mark.parametrize(
    "factor_structure",
    [
        "single_factor",
        "global_sector",
        "global_region",
        "global_sector_region",
    ],
)
def test_valid_dependence_factor_matrix(
    dependence_model,
    factor_structure,
):

    settings = create_valid_settings()

    settings[
        "dependence_model"
    ] = dependence_model

    settings[
        "factor_structure"
    ] = factor_structure

    assert validate_model_settings(
        settings
    )


# t-copula degrees of freedom must exceed two.
def test_invalid_t_degrees_of_freedom():

    settings = create_valid_settings()

    settings[
        "dependence_model"
    ] = "t_copula"

    settings[
        "t_degrees_of_freedom"
    ] = 2

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# t degrees of freedom are irrelevant for the Gaussian copula.
def test_t_degrees_of_freedom_ignored_for_gaussian_copula():

    settings = create_valid_settings()

    settings[
        "dependence_model"
    ] = "gaussian_copula"

    settings[
        "t_degrees_of_freedom"
    ] = 2

    assert validate_model_settings(
        settings
    )


# t degrees of freedom are irrelevant under independence.
def test_t_degrees_of_freedom_ignored_for_independent():

    settings = create_valid_settings()

    settings[
        "dependence_model"
    ] = "independent"

    settings[
        "t_degrees_of_freedom"
    ] = 2

    assert validate_model_settings(
        settings
    )


# Number of simulations must be positive.
def test_invalid_number_simulations():

    settings = create_valid_settings()

    settings[
        "number_simulations"
    ] = 0

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Number of simulations must be an integer.
def test_non_integer_number_simulations():

    settings = create_valid_settings()

    settings[
        "number_simulations"
    ] = 1000.5

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Seed must be non-negative.
def test_invalid_seed():

    settings = create_valid_settings()

    settings[
        "seed"
    ] = -1

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Confidence level must lie strictly between zero and one.
@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
        -0.01,
        1.01,
    ],
)
def test_invalid_confidence_level(
    confidence_level,
):

    settings = create_valid_settings()

    settings[
        "confidence_level"
    ] = confidence_level

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Factor lookback period must be positive.
def test_invalid_factor_lookback_years():

    settings = create_valid_settings()

    settings[
        "factor_lookback_years"
    ] = 0

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Supported model base currencies pass.
@pytest.mark.parametrize(
    "base_currency",
    [
        "USD",
        "EUR",
        "CHF",
        "GBP",
    ],
)
def test_supported_base_currencies(
    base_currency,
):

    settings = create_valid_settings()

    settings[
        "base_currency"
    ] = base_currency

    assert validate_model_settings(
        settings
    )


# Boolean output settings should accept True and False.
@pytest.mark.parametrize(
    "setting_name",
    [
        "create_excel_output",
        "create_pdf_output",
        "create_plot_files",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ],
)
def test_valid_boolean_output_settings(
    setting_name,
    value,
):

    settings = create_valid_settings()

    settings[
        setting_name
    ] = value

    assert validate_model_settings(
        settings
    )


# Invalid output setting values should raise an error.
@pytest.mark.parametrize(
    "setting_name",
    [
        "create_excel_output",
        "create_pdf_output",
        "create_plot_files",
    ],
)
def test_invalid_output_setting(
    setting_name,
):

    settings = create_valid_settings()

    settings[
        setting_name
    ] = "invalid"

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# output_folder remains optional.
def test_output_folder_is_optional():

    settings = create_valid_settings()

    settings.pop(
        "output_folder"
    )

    assert validate_model_settings(
        settings
    )


# An empty output folder is allowed.
def test_empty_output_folder_allowed():

    settings = create_valid_settings()

    settings[
        "output_folder"
    ] = ""

    assert validate_model_settings(
        settings
    )


# =========================================================
# Market Data Date
# =========================================================

# "latest" means that the latest available market observation is used.
def test_market_data_date_latest():

    assert parse_market_data_date(
        "latest"
    ) is None


# "latest" is case insensitive.
def test_market_data_date_latest_case_insensitive():

    assert parse_market_data_date(
        "LATEST"
    ) is None


# A valid YYYY-MM-DD string is converted into a Python date.
def test_market_data_date_specific_date():

    result = parse_market_data_date(
        "2026-08-31"
    )

    assert result == date(
        2026,
        8,
        31,
    )


# Excel date cells may be loaded as pandas Timestamp objects.
def test_market_data_date_timestamp():

    result = parse_market_data_date(
        pd.Timestamp(
            "2026-08-31"
        )
    )

    assert result == date(
        2026,
        8,
        31,
    )


# Invalid date format raises an error.
def test_invalid_market_data_date_format():

    with pytest.raises(ValueError):

        parse_market_data_date(
            "31-08-2026"
        )


# Empty market data date raises an error.
def test_empty_market_data_date():

    with pytest.raises(ValueError):

        parse_market_data_date(
            ""
        )


# Missing market data date raises an error.
def test_missing_market_data_date():

    settings = create_valid_settings()

    settings.pop(
        "market_data_date"
    )

    with pytest.raises(ValueError):

        validate_model_settings(
            settings
        )


# Future market data dates are not allowed.
def test_future_market_data_date():

    future_date = (
        date.today()
        + timedelta(days=1)
    )

    with pytest.raises(ValueError):

        parse_market_data_date(
            future_date.isoformat()
        )


# Explicit market data date passes complete settings validation.
def test_specific_market_data_date_valid_settings():

    settings = create_valid_settings()

    settings[
        "market_data_date"
    ] = "2026-08-31"

    assert validate_model_settings(
        settings
    )


# =========================================================
# Factor Proxy Validation
# =========================================================

# Valid factor proxies pass.
def test_valid_factor_proxies():

    assert validate_factor_proxies(
        create_valid_factor_proxies()
    )


# =========================================================
# Market Data Validation
# =========================================================

# Valid market data passes.
def test_valid_market_data():

    market_data = create_valid_market_data()

    assert validate_market_data(
        market_data,
        base_currency="USD",
    )


# CHF-anchored FX snapshot also supports another model base.
def test_fx_snapshot_supports_different_model_base():

    market_data = create_valid_market_data()

    assert validate_market_data(
        market_data,
        base_currency="EUR",
    )


# FX rates must be positive.
def test_invalid_fx_rate():

    market_data = create_valid_market_data()

    market_data.loc[
        1,
        "value",
    ] = 0.0

    with pytest.raises(ValueError):

        validate_market_data(
            market_data,
            base_currency="USD",
        )


# Snapshot reference currency FX rate must equal one.
def test_invalid_snapshot_reference_fx_rate():

    market_data = create_valid_market_data()

    market_data.loc[
        0,
        "value",
    ] = 0.99

    with pytest.raises(ValueError):

        validate_market_data(
            market_data,
            base_currency="USD",
        )


# Negative risk-free rates are allowed.
def test_negative_risk_free_rate_allowed():

    market_data = create_valid_market_data()

    market_data.loc[
        4,
        "value",
    ] = -0.005

    assert validate_market_data(
        market_data,
        base_currency="USD",
    )


# Duplicate risk-free curve point raises an error.
def test_duplicate_risk_free_curve_point():

    market_data = create_valid_market_data()

    duplicate_row = (
        market_data
        .iloc[[5]]
        .copy()
    )

    market_data = pd.concat(
        [
            market_data,
            duplicate_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):

        validate_market_data(
            market_data,
            base_currency="USD",
        )


# =========================================================
# Rating Migration Matrix Validation
# =========================================================

# Valid rating migration matrix passes.
def test_valid_rating_migration_matrix():

    migration_matrix = (
        create_valid_rating_migration_matrix()
    )

    assert validate_rating_migration_matrix(
        migration_matrix
    )