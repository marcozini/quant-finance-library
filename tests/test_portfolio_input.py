#### Unit Tests for Credit Portfolio Input ####

import pandas as pd
import pytest

from quant_finance.credit.portfolio_input import (
    validate_portfolio,
    validate_model_settings,
    validate_factor_proxies,
    validate_rating_migration_matrix,
)


# Create a small valid portfolio for validation tests.

def create_valid_portfolio():

    return pd.DataFrame({
        "counterparty": [
            "Company A",
            "Company B",
        ],
        "ticker": [
            None,
            "AAPL",
        ],
        "factor_loading_proxy_ticker": [
            "SIE.DE",
            None,
        ],
        "exposure": [
            10_000_000,
            5_000_000,
        ],
        "currency": [
            "CHF",
            "USD",
        ],
        "lgd": [
            0.45,
            0.40,
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
            None,
        ],
        "equity_volatility": [
            None,
            None,
        ],
        "debt": [
            None,
            None,
        ],
        "risk_free_rate": [
            None,
            None,
        ],
        "maturity": [
            5,
            3,
        ],
        "coupon_rate": [
            0.03,
            0.04,
        ],
        "payment_frequency": [
            2,
            1,
        ],
        "sector": [
            "Financials",
            "Technology",
        ],
        "region": [
            "Europe",
            "US",
        ],
    })


# Create valid model settings for validation tests.

def create_valid_model_settings():

    return {
        "number_simulations": 100000,
        "seed": 0,
        "confidence_level": 0.995,
        "dependence_model": "t_copula",
        "t_degrees_of_freedom": 5,
        "factor_structure": "global_sector_region",
        "factor_lookback_years": 3,
    }


# Create a valid synthetic rating migration matrix.

def create_valid_rating_migration_matrix():

    return pd.DataFrame({
        "current_rating": [
            "AAA",
            "AA",
            "A",
            "BBB",
            "BB",
            "B",
            "CCC",
            "D",
        ],
        "spread_bps": [
            30,
            50,
            80,
            150,
            300,
            600,
            1200,
            None,
        ],
        "spread_decimal": [
            0.003,
            0.005,
            0.008,
            0.015,
            0.030,
            0.060,
            0.120,
            None,
        ],
        "AAA": [
            0.9150,
            0.0080,
            0.0010,
            0.0002,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        "AA": [
            0.0750,
            0.9000,
            0.0250,
            0.0030,
            0.0005,
            0.0001,
            0.0000,
            0.0000,
        ],
        "A": [
            0.0080,
            0.0750,
            0.8900,
            0.0350,
            0.0050,
            0.0010,
            0.0005,
            0.0000,
        ],
        "BBB": [
            0.0010,
            0.0120,
            0.0650,
            0.8500,
            0.0350,
            0.0050,
            0.0015,
            0.0000,
        ],
        "BB": [
            0.0003,
            0.0020,
            0.0120,
            0.0750,
            0.8000,
            0.0500,
            0.0080,
            0.0000,
        ],
        "B": [
            0.0001,
            0.0010,
            0.0030,
            0.0200,
            0.1000,
            0.7800,
            0.0500,
            0.0000,
        ],
        "CCC": [
            0.0001,
            0.0005,
            0.0010,
            0.0050,
            0.0250,
            0.0800,
            0.6000,
            0.0000,
        ],
        "D": [
            0.0005,
            0.0015,
            0.0030,
            0.0118,
            0.0345,
            0.0839,
            0.3400,
            1.0000,
        ],
        "row_sum": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    })


# 1. Valid portfolio passes validation.

def test_valid_portfolio():

    portfolio = create_valid_portfolio()

    assert validate_portfolio(
        portfolio
    ) is True


# 2. Negative exposure raises an error.

def test_negative_exposure():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "exposure",
    ] = -1

    with pytest.raises(ValueError):
        validate_portfolio(
            portfolio
        )


# 3. Invalid LGD raises an error.

def test_invalid_lgd():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "lgd",
    ] = 1.2

    with pytest.raises(ValueError):
        validate_portfolio(
            portfolio
        )


# 4. Invalid PD method raises an error.

def test_invalid_pd_method():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "pd_method",
    ] = "other"

    with pytest.raises(ValueError):
        validate_portfolio(
            portfolio
        )


# 5. Rating method requires a rating.

def test_missing_rating():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "rating",
    ] = None

    with pytest.raises(ValueError):
        validate_portfolio(
            portfolio
        )


# 6. Merton method requires either ticker or manual inputs.

def test_missing_merton_inputs():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        1,
        "ticker",
    ] = None

    with pytest.raises(ValueError):
        validate_portfolio(
            portfolio
        )


# 7. Valid model settings pass validation.

def test_valid_model_settings():

    settings = create_valid_model_settings()

    assert validate_model_settings(
        settings
    ) is True


# 8. Invalid dependence model raises an error.

def test_invalid_dependence_model():

    settings = create_valid_model_settings()

    settings[
        "dependence_model"
    ] = "invalid"

    with pytest.raises(ValueError):
        validate_model_settings(
            settings
        )


# 9. Invalid t-copula degrees of freedom raises an error.

def test_invalid_t_degrees_of_freedom():

    settings = create_valid_model_settings()

    settings[
        "t_degrees_of_freedom"
    ] = 2

    with pytest.raises(ValueError):
        validate_model_settings(
            settings
        )


# 10. Valid factor proxies pass validation.

def test_valid_factor_proxies():

    factor_proxies = pd.DataFrame({
        "factor_type": [
            "global",
            "region",
            "sector",
        ],
        "factor_name": [
            "Global",
            "Europe",
            "Industrial",
        ],
        "ticker": [
            "ACWI",
            "VGK",
            "EXI",
        ],
        "description": [
            "Global proxy",
            "Europe proxy",
            "Industrial proxy",
        ],
    })

    assert validate_factor_proxies(
        factor_proxies
    ) is True


# 11. Duplicate factor mapping raises an error.

def test_duplicate_factor_proxy():

    factor_proxies = pd.DataFrame({
        "factor_type": [
            "global",
            "region",
            "region",
        ],
        "factor_name": [
            "Global",
            "Europe",
            "Europe",
        ],
        "ticker": [
            "ACWI",
            "VGK",
            "VGK",
        ],
        "description": [
            "Global proxy",
            "Europe proxy",
            "Europe proxy",
        ],
    })

    with pytest.raises(ValueError):
        validate_factor_proxies(
            factor_proxies
        )


# 12. Unlisted counterparty requires a factor-loading proxy ticker.

def test_missing_factor_loading_proxy():

    portfolio = create_valid_portfolio()

    portfolio.loc[
        0,
        "factor_loading_proxy_ticker",
    ] = None

    with pytest.raises(ValueError):
        validate_portfolio(
            portfolio
        )


# 13. Valid rating migration matrix passes validation.

def test_valid_rating_migration_matrix():

    rating_migration_matrix = (
        create_valid_rating_migration_matrix()
    )

    assert validate_rating_migration_matrix(
        rating_migration_matrix
    ) is True


# 14. Rating transition rows must sum to one.

def test_invalid_rating_transition_row_sum():

    rating_migration_matrix = (
        create_valid_rating_migration_matrix()
    )

    rating_migration_matrix.loc[
        0,
        "AAA",
    ] = 0.80

    with pytest.raises(ValueError):
        validate_rating_migration_matrix(
            rating_migration_matrix
        )