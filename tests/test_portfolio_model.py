#### Unit Tests Credit Portfolio Model ####

import numpy as np
import pandas as pd

import quant_finance.credit.portfolio_model as portfolio_model


def test_run_credit_portfolio_model(monkeypatch):

    # Minimal synthetic portfolio.
    portfolio = pd.DataFrame({
        "counterparty": ["Test Co"],
        "exposure": [100.0],
        "lgd": [0.40],
        "pd": [0.01],
        "expected_loss": [0.40],
    })

    settings = {
        "confidence_level": 0.75,
        "factor_lookback_years": 3,
        "base_currency": "USD",
    }

    factor_proxies = pd.DataFrame()
    rating_migration_matrix = pd.DataFrame()
    market_data = pd.DataFrame()

    portfolio_losses = np.array([
        0.0,
        0.0,
        5.0,
        10.0,
    ])

    counterparty_losses = (
        portfolio_losses.reshape(-1, 1)
    )

    migration_states = np.array(
        [
            ["A"],
            ["A"],
            ["BBB"],
            ["D"],
        ],
        dtype=object,
    )

    # Mock Excel loading.
    monkeypatch.setattr(
        portfolio_model,
        "load_portfolio",
        lambda _: portfolio.copy(),
    )

    monkeypatch.setattr(
        portfolio_model,
        "load_model_settings",
        lambda _: settings,
    )

    monkeypatch.setattr(
        portfolio_model,
        "load_factor_proxies",
        lambda _: factor_proxies,
    )

    monkeypatch.setattr(
        portfolio_model,
        "load_rating_migration_matrix",
        lambda _: rating_migration_matrix,
    )

    monkeypatch.setattr(
        portfolio_model,
        "load_market_data",
        lambda _: market_data,
    )

    # Validation functions are tested separately.
    monkeypatch.setattr(
        portfolio_model,
        "validate_portfolio",
        lambda _: None,
    )

    monkeypatch.setattr(
        portfolio_model,
        "validate_model_settings",
        lambda _: None,
    )

    monkeypatch.setattr(
        portfolio_model,
        "validate_factor_proxies",
        lambda _: None,
    )

    monkeypatch.setattr(
        portfolio_model,
        "validate_rating_migration_matrix",
        lambda _: None,
    )

    monkeypatch.setattr(
        portfolio_model,
        "validate_market_data",
        lambda market_data, base_currency: None,
    )

    # Components below have dedicated tests.
    monkeypatch.setattr(
        portfolio_model,
        "_complete_risk_free_rates",
        lambda portfolio, market_data: portfolio,
    )

    monkeypatch.setattr(
        portfolio_model,
        "_convert_exposures_to_base_currency",
        lambda portfolio, base_currency, market_data: portfolio,
    )

    monkeypatch.setattr(
        portfolio_model,
        "_calculate_counterparty_pds",
        lambda portfolio, **kwargs: portfolio,
    )

    monkeypatch.setattr(
        portfolio_model,
        "_calculate_expected_losses",
        lambda portfolio: portfolio,
    )

    monkeypatch.setattr(
        portfolio_model,
        "_estimate_portfolio_factor_loadings",
        lambda portfolio, **kwargs: portfolio,
    )

    monkeypatch.setattr(
        portfolio_model,
        "simulate_migration_portfolio_losses",
        lambda **kwargs: (
            portfolio_losses,
            migration_states,
            counterparty_losses,
        ),
    )

    results = (
        portfolio_model
        .run_credit_portfolio_model(
            "dummy.xlsx"
        )
    )

    # Check main output structure.
    assert "portfolio" in results
    assert "settings" in results
    assert "market_data" in results
    assert "summary" in results
    assert "portfolio_losses" in results
    assert "migration_states" in results
    assert "counterparty_losses" in results

    # Check selected model base currency.
    assert (
        results["summary"]["base_currency"]
        == "USD"
    )

    # Check portfolio risk metrics.
    assert (
        "default_expected_loss"
        in results["summary"]
    )

    assert (
        "migration_mean_loss"
        in results["summary"]
    )

    assert (
        "value_at_risk"
        in results["summary"]
    )

    assert (
        "expected_shortfall"
        in results["summary"]
    )

    assert (
        "unexpected_loss"
        in results["summary"]
    )

    # Check risk contribution columns.
    assert (
        "incremental_var"
        in results["portfolio"].columns
    )

    assert (
        "incremental_es"
        in results["portfolio"].columns
    )

    assert (
        "marginal_var"
        in results["portfolio"].columns
    )

    assert (
        "marginal_es"
        in results["portfolio"].columns
    )


# Listed counterparty falls back to proxy ticker
# if its own factor-market-data retrieval fails.
def test_factor_loading_fallback_proxy(monkeypatch):

    portfolio = pd.DataFrame({
        "counterparty": ["Test Co"],
        "ticker": ["BAD"],
        "factor_loading_proxy_ticker": ["PROXY"],
        "region": ["US"],
        "sector": ["Technology"],
    })

    def mock_factor_loadings(
        company_ticker,
        **kwargs,
    ):

        if company_ticker == "BAD":
            raise ValueError(
                "No market data found."
            )

        return (
            0.40,
            0.20,
            0.10,
            0.80,
        )

    monkeypatch.setattr(
        portfolio_model,
        "estimate_counterparty_factor_loadings",
        mock_factor_loadings,
    )

    result = (
        portfolio_model
        ._estimate_portfolio_factor_loadings(
            portfolio=portfolio,
            factor_proxies=pd.DataFrame(),
            start_date=None,
            end_date=None,
        )
    )

    assert (
        result.loc[
            0,
            "factor_calibration_ticker",
        ]
        == "PROXY"
    )

    assert (
        result.loc[
            0,
            "factor_source",
        ]
        == "fallback_proxy_ticker"
    )

    assert (
        result.loc[
            0,
            "global_loading",
        ]
        == 0.40
    )

    assert (
        result.loc[
            0,
            "region_loading",
        ]
        == 0.20
    )

    assert (
        result.loc[
            0,
            "sector_loading",
        ]
        == 0.10
    )

    assert (
        result.loc[
            0,
            "idiosyncratic_loading",
        ]
        == 0.80
    )