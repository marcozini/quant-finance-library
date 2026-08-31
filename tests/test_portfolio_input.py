#### Unit Tests Credit Portfolio Input ####

import pandas as pd
import pytest

from quant_finance.credit.portfolio_input import (
    validate_portfolio,
    validate_model_settings,
    validate_factor_proxies,
    validate_market_data,
    validate_rating_migration_matrix,
)


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
        [0.90, 0.08, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0005],
        [0.02, 0.90, 0.06, 0.01, 0.005, 0.002, 0.001, 0.002],
        [0.005, 0.03, 0.88, 0.06, 0.015, 0.005, 0.002, 0.003],
        [0.002, 0.008, 0.04, 0.84, 0.07, 0.025, 0.008, 0.007],
        [0.001, 0.003, 0.01, 0.05, 0.80, 0.09, 0.025, 0.021],
        [0.001, 0.001, 0.003, 0.01, 0.06, 0.76, 0.09, 0.075],
        [0.0005, 0.0005, 0.001, 0.003, 0.01, 0.05, 0.70, 0.235],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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


# 1. Valid portfolio passes.
def test_valid_portfolio():
    assert validate_portfolio(create_valid_portfolio())


# 2. risk_free_rate is no longer required in the portfolio input.
def test_portfolio_does_not_require_risk_free_rate():
    portfolio = create_valid_portfolio()

    assert "risk_free_rate" not in portfolio.columns
    assert validate_portfolio(portfolio)


# 3. Missing required portfolio column raises an error.
def test_missing_portfolio_column():
    portfolio = create_valid_portfolio().drop(columns=["exposure"])

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 4. Exposure must be positive.
def test_invalid_exposure():
    portfolio = create_valid_portfolio()
    portfolio.loc[0, "exposure"] = 0

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 5. LGD must lie between zero and one.
def test_invalid_lgd():
    portfolio = create_valid_portfolio()
    portfolio.loc[0, "lgd"] = 1.20

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 6. Currency is required.
def test_missing_currency():
    portfolio = create_valid_portfolio()
    portfolio.loc[0, "currency"] = None

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 7. PD method must be valid.
def test_invalid_pd_method():
    portfolio = create_valid_portfolio()
    portfolio.loc[0, "pd_method"] = "invalid"

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 8. Rating-based counterparties require a rating.
def test_rating_method_requires_rating():
    portfolio = create_valid_portfolio()
    portfolio.loc[0, "rating"] = None

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 9. Manual Merton inputs work without a risk-free-rate input.
def test_manual_merton_inputs_without_risk_free_rate():
    portfolio = create_valid_portfolio()

    assert validate_portfolio(portfolio)


# 10. Incomplete Merton inputs without ticker raise an error.
def test_incomplete_manual_merton_inputs():
    portfolio = create_valid_portfolio()
    portfolio.loc[1, "equity_value"] = None

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 11. Unlisted counterparty requires a factor proxy ticker.
def test_unlisted_counterparty_requires_factor_proxy():
    portfolio = create_valid_portfolio()
    portfolio.loc[1, "factor_loading_proxy_ticker"] = None

    with pytest.raises(ValueError):
        validate_portfolio(portfolio)


# 12. Valid model settings pass.
def test_valid_model_settings():
    assert validate_model_settings(create_valid_settings())


# 13. Invalid base currency raises an error.
def test_invalid_base_currency():
    settings = create_valid_settings()
    settings["base_currency"] = "US"

    with pytest.raises(ValueError):
        validate_model_settings(settings)


# 14. Valid factor proxies pass.
def test_valid_factor_proxies():
    assert validate_factor_proxies(create_valid_factor_proxies())


# 15. Valid market data passes.
def test_valid_market_data():
    market_data = create_valid_market_data()

    assert validate_market_data(
        market_data,
        base_currency="USD",
    )


# 16. CHF-anchored FX snapshot also supports another model base.
def test_fx_snapshot_supports_different_model_base():
    market_data = create_valid_market_data()

    assert validate_market_data(
        market_data,
        base_currency="EUR",
    )


# 17. FX rates must be positive.
def test_invalid_fx_rate():
    market_data = create_valid_market_data()
    market_data.loc[1, "value"] = 0.0

    with pytest.raises(ValueError):
        validate_market_data(
            market_data,
            base_currency="USD",
        )


# 18. Snapshot reference currency FX rate must equal one.
def test_invalid_snapshot_reference_fx_rate():
    market_data = create_valid_market_data()
    market_data.loc[0, "value"] = 0.99

    with pytest.raises(ValueError):
        validate_market_data(
            market_data,
            base_currency="USD",
        )


# 19. Negative risk-free rates are allowed.
def test_negative_risk_free_rate_allowed():
    market_data = create_valid_market_data()
    market_data.loc[4, "value"] = -0.005

    assert validate_market_data(
        market_data,
        base_currency="USD",
    )


# 20. Duplicate risk-free curve point raises an error.
def test_duplicate_risk_free_curve_point():
    market_data = create_valid_market_data()

    duplicate_row = market_data.iloc[[5]].copy()

    market_data = pd.concat(
        [market_data, duplicate_row],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        validate_market_data(
            market_data,
            base_currency="USD",
        )


# 21. Valid rating migration matrix passes.
def test_valid_rating_migration_matrix():
    migration_matrix = create_valid_rating_migration_matrix()

    assert validate_rating_migration_matrix(
        migration_matrix
    )