#### Unit Tests for Credit Portfolio Input ####

import pandas as pd
import pytest

from quant_finance.credit.portfolio_input import (
    validate_portfolio,
    validate_model_settings,
    validate_factor_proxies,
    validate_market_data,
    validate_rating_migration_matrix,
)


# Create a small valid portfolio for validation tests.

def create_valid_portfolio():

    return pd.DataFrame({
        "counterparty": ["Company A", "Company B"],
        "ticker": [None, "AAPL"],
        "factor_loading_proxy_ticker": ["SIE.DE", None],
        "exposure": [10_000_000, 5_000_000],
        "currency": ["CHF", "USD"],
        "lgd": [0.45, 0.40],
        "pd_method": ["rating", "merton"],
        "rating": ["AA", None],
        "equity_value": [None, None],
        "equity_volatility": [None, None],
        "debt": [None, None],
        "risk_free_rate": [None, None],
        "maturity": [5, 3],
        "coupon_rate": [0.03, 0.04],
        "payment_frequency": [2, 1],
        "sector": ["Financials", "Technology"],
        "region": ["Europe", "US"],
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
        "base_currency": "CHF",
    }


# Create valid fallback market data.

def create_valid_market_data():

    return pd.DataFrame({
        "data_type": [
            "fx",
            "fx",
            "risk_free",
            "risk_free",
        ],
        "currency": [
            "CHF",
            "EUR",
            "CHF",
            "EUR",
        ],
        "base_currency": [
            "CHF",
            "CHF",
            None,
            None,
        ],
        "tenor_years": [
            None,
            None,
            1.0,
            1.0,
        ],
        "value": [
            1.0,
            0.93,
            0.00,
            0.02,
        ],
        "as_of_date": [
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
            "2025-12-31",
        ],
        "curve_type": [
            None,
            None,
            "flat_policy_proxy",
            "flat_policy_proxy",
        ],
        "source": [
            "ECB",
            "ECB",
            "SNB",
            "ECB",
        ],
    })


# Create a valid synthetic rating migration matrix.

def create_valid_rating_migration_matrix():

    return pd.DataFrame({
        "current_rating": [
            "AAA", "AA", "A", "BBB",
            "BB", "B", "CCC", "D",
        ],
        "spread_bps": [
            30, 50, 80, 150,
            300, 600, 1200, None,
        ],
        "spread_decimal": [
            0.003, 0.005, 0.008, 0.015,
            0.030, 0.060, 0.120, None,
        ],
        "AAA": [
            0.9150, 0.0080, 0.0010, 0.0002,
            0.0000, 0.0000, 0.0000, 0.0000,
        ],
        "AA": [
            0.0750, 0.9000, 0.0250, 0.0030,
            0.0005, 0.0001, 0.0000, 0.0000,
        ],
        "A": [
            0.0080, 0.0750, 0.8900, 0.0350,
            0.0050, 0.0010, 0.0005, 0.0000,
        ],
        "BBB": [
            0.0010, 0.0120, 0.0650, 0.8500,
            0.0350, 0.0050, 0.0015, 0.0000,
        ],
        "BB": [
            0.0003, 0.0020, 0.0120, 0.0750,
            0.8000, 0.0500, 0.0080, 0.0000,
        ],
        "B": [
            0.0001, 0.0010, 0.0030, 0.0200,
            0.1000, 0.7800, 0.0500, 0.0000,
        ],
        "CCC": [
            0.0001, 0.0005, 0.0010, 0.0050,
            0.0250, 0.0800, 0.6000, 0.0000,
        ],
        "D": [
            0.0005, 0.0015, 0.0030, 0.0118,
            0.0345, 0.0839, 0.3400, 1.0000,
        ],
        "row_sum": [
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ],
    })


# 1. Valid portfolio passes validation.

def test_valid_portfolio():

    portfolio = create_valid_portfolio()

    assert validate_portfolio(portfolio) is True


# 2. Negative exposure raises an error.

def test_negative_exposure():

    portfolio = create_valid_portfolio()
    portfolio.loc[0, "exposure"] = -1

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 3. Invalid LGD raises an error.

def test_invalid_lgd():

    portfolio = create_valid_portfolio()
    portfolio.loc[0, "lgd"] = 1.2

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 4. Invalid PD method raises an error.

def test_invalid_pd_method():

    portfolio = create_valid_portfolio()
    portfolio.loc[0, "pd_method"] = "other"

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 5. Rating method requires a rating.

def test_missing_rating():

    portfolio = create_valid_portfolio()
    portfolio.loc[0, "rating"] = None

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 6. Merton method requires either ticker or manual inputs.

def test_missing_merton_inputs():

    portfolio = create_valid_portfolio()
    portfolio.loc[1, "ticker"] = None

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 7. Manual Merton inputs do not require a manual risk-free rate.

def test_manual_merton_without_risk_free_rate():

    portfolio = create_valid_portfolio()

    portfolio.loc[1, "ticker"] = None
    portfolio.loc[1, "factor_loading_proxy_ticker"] = "AAPL"
    portfolio.loc[1, "equity_value"] = 100.0
    portfolio.loc[1, "equity_volatility"] = 0.25
    portfolio.loc[1, "debt"] = 50.0
    portfolio.loc[1, "risk_free_rate"] = None

    assert validate_portfolio(portfolio) is True


# 8. Valid model settings pass validation.

def test_valid_model_settings():

    settings = create_valid_model_settings()

    assert validate_model_settings(settings) is True


# 9. Invalid dependence model raises an error.

def test_invalid_dependence_model():

    settings = create_valid_model_settings()
    settings["dependence_model"] = "invalid"

    with pytest.raises(ValueError):
        validate_model_settings(settings)


# 10. Invalid t-copula degrees of freedom raises an error.

def test_invalid_t_degrees_of_freedom():

    settings = create_valid_model_settings()
    settings["t_degrees_of_freedom"] = 2

    with pytest.raises(ValueError):
        validate_model_settings(settings)


# 11. Invalid base currency raises an error.

def test_invalid_base_currency():

    settings = create_valid_model_settings()
    settings["base_currency"] = "CH"

    with pytest.raises(ValueError):
        validate_model_settings(settings)


# 12. Valid factor proxies pass validation.

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

    assert validate_factor_proxies(factor_proxies) is True


# 13. Duplicate factor mapping raises an error.

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
        validate_factor_proxies(factor_proxies)


# 14. Unlisted counterparty requires a factor-loading proxy ticker.

def test_missing_factor_loading_proxy():

    portfolio = create_valid_portfolio()
    portfolio.loc[0, "factor_loading_proxy_ticker"] = None

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 15. Valid rating migration matrix passes validation.

def test_valid_rating_migration_matrix():

    rating_migration_matrix = (
        create_valid_rating_migration_matrix()
    )

    assert validate_rating_migration_matrix(
        rating_migration_matrix
    ) is True


# 16. Rating transition rows must sum to one.

def test_invalid_rating_transition_row_sum():

    rating_migration_matrix = (
        create_valid_rating_migration_matrix()
    )

    rating_migration_matrix.loc[0, "AAA"] = 0.80

    with pytest.raises(ValueError):
        validate_rating_migration_matrix(
            rating_migration_matrix
        )


# 17. Valid fallback market data passes validation.

def test_valid_market_data():

    market_data = create_valid_market_data()

    assert validate_market_data(
        market_data,
        base_currency="CHF",
    ) is True


# 18. FX fallback rates must be positive.

def test_invalid_fx_rate():

    market_data = create_valid_market_data()
    market_data.loc[1, "value"] = 0.0

    with pytest.raises(ValueError):
        validate_market_data(
            market_data,
            base_currency="CHF",
        )


# 19. Base-currency FX rate must equal one.

def test_invalid_base_currency_fx_rate():

    market_data = create_valid_market_data()
    market_data.loc[0, "value"] = 0.99

    with pytest.raises(ValueError):
        validate_market_data(
            market_data,
            base_currency="CHF",
        )


# 20. Negative risk-free fallback rates are allowed.

def test_negative_risk_free_rate_allowed():

    market_data = create_valid_market_data()
    market_data.loc[2, "value"] = -0.005

    assert validate_market_data(
        market_data,
        base_currency="CHF",
    ) is True


# 21. Duplicate risk-free curve points raise an error.

def test_duplicate_risk_free_curve_point():

    market_data = create_valid_market_data()

    duplicate_row = market_data.loc[[2]].copy()

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
            base_currency="CHF",
        )