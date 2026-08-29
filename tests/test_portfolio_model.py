#### Unit Test Credit Portfolio Model ####

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
    }

    factor_proxies = pd.DataFrame()
    rating_migration_matrix = pd.DataFrame()

    portfolio_losses = np.array([0.0, 0.0, 5.0, 10.0])
    counterparty_losses = portfolio_losses.reshape(-1, 1)
    migration_states = np.array(
        [["A"], ["A"], ["BBB"], ["D"]],
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

    # Mock components that already have dedicated tests.
    monkeypatch.setattr(
        portfolio_model,
        "_complete_risk_free_rates",
        lambda portfolio: portfolio,
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

    results = portfolio_model.run_credit_portfolio_model(
        "dummy.xlsx"
    )

    # Check main output structure.
    assert "portfolio" in results
    assert "summary" in results
    assert "portfolio_losses" in results
    assert "migration_states" in results
    assert "counterparty_losses" in results

    # Check portfolio risk metrics.
    assert "default_expected_loss" in results["summary"]
    assert "migration_mean_loss" in results["summary"]
    assert "value_at_risk" in results["summary"]
    assert "expected_shortfall" in results["summary"]
    assert "unexpected_loss" in results["summary"]

    # Check risk contribution columns.
    assert "incremental_var" in results["portfolio"].columns
    assert "incremental_es" in results["portfolio"].columns
    assert "marginal_var" in results["portfolio"].columns
    assert "marginal_es" in results["portfolio"].columns