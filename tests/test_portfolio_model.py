#### Unit Tests Credit Portfolio Model ####

import numpy as np
import pandas as pd

import quant_finance.credit.portfolio_model as portfolio_model


# ---------------------------------------------------------
# Complete model orchestration
# ---------------------------------------------------------

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
    risk_columns = [
        "incremental_var",
        "incremental_es",
        "marginal_var",
        "marginal_es",
        "standalone_var",
        "standalone_es",
        "var_diversification_benefit",
        "es_diversification_benefit",
    ]

    for column in risk_columns:

        assert (
            column
            in results["portfolio"].columns
        )


# ---------------------------------------------------------
# Counterparty PD calculation
# ---------------------------------------------------------

# Rating-based counterparty uses rating PD.
def test_rating_based_counterparty_pd(monkeypatch):

    portfolio = pd.DataFrame({
        "counterparty": ["Rating Co"],
        "pd_method": ["rating"],
        "rating": ["AA"],
    })

    monkeypatch.setattr(
        portfolio_model,
        "get_counterparty_pd",
        lambda **kwargs: 0.001,
    )

    result = (
        portfolio_model
        ._calculate_counterparty_pds(
            portfolio=portfolio,
            start_date=None,
            end_date=None,
        )
    )

    assert np.isclose(
        result.loc[0, "pd"],
        0.001,
    )

    assert (
        result.loc[
            0,
            "pd_source",
        ]
        == "rating"
    )


# Fully manual Merton inputs are used without market-data retrieval.
def test_manual_merton_counterparty_pd(monkeypatch):

    portfolio = pd.DataFrame({
        "counterparty": ["Private Co"],
        "ticker": [None],
        "pd_method": ["merton"],
        "rating": [None],
        "equity_value": [20.0],
        "equity_volatility": [0.45],
        "debt": [80.0],
        "risk_free_rate": [0.03],
    })

    # Manual inputs should mean these functions are never needed.
    monkeypatch.setattr(
        portfolio_model,
        "estimate_equity_value",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "Equity market data should not be requested."
                )
            )
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "estimate_equity_volatility",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "Volatility market data should not be requested."
                )
            )
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "estimate_debt_default_point",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "Debt market data should not be requested."
                )
            )
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "calibrate_merton",
        lambda *args, **kwargs: (
            120.0,
            0.25,
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "merton_probability_of_default",
        lambda *args, **kwargs: 0.012,
    )

    result = (
        portfolio_model
        ._calculate_counterparty_pds(
            portfolio=portfolio,
            start_date=None,
            end_date=None,
        )
    )

    assert np.isclose(
        result.loc[
            0,
            "pd",
        ],
        0.012,
    )

    assert (
        result.loc[
            0,
            "pd_source",
        ]
        == "merton_manual"
    )

    assert np.isclose(
        result.loc[
            0,
            "asset_value",
        ],
        120.0,
    )

    assert np.isclose(
        result.loc[
            0,
            "asset_volatility",
        ],
        0.25,
    )


# Listed Merton counterparty retrieves missing company data.
def test_market_data_merton_counterparty_pd(monkeypatch):

    portfolio = pd.DataFrame({
        "counterparty": ["Listed Co"],
        "ticker": ["TEST"],
        "pd_method": ["merton"],
        "rating": [None],
        "equity_value": [None],
        "equity_volatility": [None],
        "debt": [None],
        "risk_free_rate": [0.03],
    })

    monkeypatch.setattr(
        portfolio_model,
        "estimate_equity_value",
        lambda ticker: 100.0,
    )

    monkeypatch.setattr(
        portfolio_model,
        "estimate_equity_volatility",
        lambda ticker, start_date, end_date: 0.30,
    )

    monkeypatch.setattr(
        portfolio_model,
        "estimate_debt_default_point",
        lambda ticker: (
            60.0,
            40.0,
            40.0,
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "calibrate_merton",
        lambda *args, **kwargs: (
            170.0,
            0.20,
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "merton_probability_of_default",
        lambda *args, **kwargs: 0.004,
    )

    result = (
        portfolio_model
        ._calculate_counterparty_pds(
            portfolio=portfolio,
            start_date=None,
            end_date=None,
        )
    )

    assert np.isclose(
        result.loc[
            0,
            "equity_value",
        ],
        100.0,
    )

    assert np.isclose(
        result.loc[
            0,
            "equity_volatility",
        ],
        0.30,
    )

    assert np.isclose(
        result.loc[
            0,
            "debt",
        ],
        60.0,
    )

    assert np.isclose(
        result.loc[
            0,
            "pd",
        ],
        0.004,
    )

    assert (
        result.loc[
            0,
            "pd_source",
        ]
        == "merton_market_data"
    )


# Partially manual Merton inputs are combined with market data.
def test_mixed_merton_counterparty_pd(monkeypatch):

    portfolio = pd.DataFrame({
        "counterparty": ["Mixed Co"],
        "ticker": ["TEST"],
        "pd_method": ["merton"],
        "rating": [None],
        "equity_value": [100.0],
        "equity_volatility": [None],
        "debt": [None],
        "risk_free_rate": [0.03],
    })

    # Equity value is already supplied manually.
    monkeypatch.setattr(
        portfolio_model,
        "estimate_equity_value",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "Manual equity value should be retained."
                )
            )
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "estimate_equity_volatility",
        lambda ticker, start_date, end_date: 0.30,
    )

    monkeypatch.setattr(
        portfolio_model,
        "estimate_debt_default_point",
        lambda ticker: (
            60.0,
            40.0,
            40.0,
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "calibrate_merton",
        lambda *args, **kwargs: (
            170.0,
            0.20,
        ),
    )

    monkeypatch.setattr(
        portfolio_model,
        "merton_probability_of_default",
        lambda *args, **kwargs: 0.005,
    )

    result = (
        portfolio_model
        ._calculate_counterparty_pds(
            portfolio=portfolio,
            start_date=None,
            end_date=None,
        )
    )

    assert np.isclose(
        result.loc[
            0,
            "equity_value",
        ],
        100.0,
    )

    assert np.isclose(
        result.loc[
            0,
            "equity_volatility",
        ],
        0.30,
    )

    assert np.isclose(
        result.loc[
            0,
            "debt",
        ],
        60.0,
    )

    assert (
        result.loc[
            0,
            "pd_source",
        ]
        == "merton_mixed"
    )


# ---------------------------------------------------------
# Factor-loading ticker logic
# ---------------------------------------------------------

# Listed counterparty uses its own ticker when retrieval succeeds.
def test_factor_loading_uses_own_ticker(monkeypatch):

    portfolio = pd.DataFrame({
        "counterparty": ["Listed Co"],
        "ticker": ["TEST"],
        "factor_loading_proxy_ticker": ["PROXY"],
        "region": ["US"],
        "sector": ["Technology"],
    })

    monkeypatch.setattr(
        portfolio_model,
        "estimate_counterparty_factor_loadings",
        lambda company_ticker, **kwargs: (
            0.40,
            0.20,
            0.10,
            0.80,
        ),
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
        == "TEST"
    )

    assert (
        result.loc[
            0,
            "factor_source",
        ]
        == "own_ticker"
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


# Unlisted counterparty uses its proxy ticker directly.
def test_unlisted_counterparty_uses_proxy_ticker(monkeypatch):

    portfolio = pd.DataFrame({
        "counterparty": ["Private Co"],
        "ticker": [None],
        "factor_loading_proxy_ticker": ["PROXY"],
        "region": ["Europe"],
        "sector": ["Industrial"],
    })

    used_tickers = []

    def mock_factor_loadings(
        company_ticker,
        **kwargs,
    ):

        used_tickers.append(
            company_ticker
        )

        return (
            0.35,
            0.15,
            0.20,
            0.85,
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
        used_tickers
        == ["PROXY"]
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
        == "proxy_ticker"
    )


# ---------------------------------------------------------
# Risk contribution consistency
# ---------------------------------------------------------

# Risk contribution calculations satisfy the model identities.
def test_risk_contribution_consistency():

    portfolio = pd.DataFrame({
        "counterparty": [
            "Counterparty A",
            "Counterparty B",
        ],
        "exposure": [
            100.0,
            50.0,
        ],
    })

    # Same Monte Carlo scenarios for both counterparties.
    counterparty_losses = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [5.0, 0.0],
        [5.0, 3.0],
        [10.0, 5.0],
        [20.0, 10.0],
    ])

    portfolio_losses = (
        counterparty_losses.sum(
            axis=1
        )
    )

    confidence_level = 0.80

    portfolio_var = (
        portfolio_model
        .value_at_risk(
            portfolio_losses,
            confidence_level,
        )
    )

    portfolio_es = (
        portfolio_model
        .expected_shortfall(
            portfolio_losses,
            confidence_level,
        )
    )

    result = (
        portfolio_model
        ._calculate_risk_contributions(
            portfolio=portfolio,
            portfolio_losses=portfolio_losses,
            counterparty_losses=counterparty_losses,
            confidence_level=confidence_level,
            portfolio_var=portfolio_var,
            portfolio_es=portfolio_es,
        )
    )

    for index in result.index:

        exposure = result.loc[
            index,
            "exposure",
        ]

        # Marginal risk is exposure-normalized
        # leave-one-out incremental risk.
        assert np.isclose(
            result.loc[
                index,
                "marginal_var",
            ],
            result.loc[
                index,
                "incremental_var",
            ]
            / exposure,
        )

        assert np.isclose(
            result.loc[
                index,
                "marginal_es",
            ],
            result.loc[
                index,
                "incremental_es",
            ]
            / exposure,
        )

        # Diversification benefit equals standalone
        # risk minus incremental portfolio risk.
        assert np.isclose(
            result.loc[
                index,
                "var_diversification_benefit",
            ],
            result.loc[
                index,
                "standalone_var",
            ]
            - result.loc[
                index,
                "incremental_var",
            ],
        )

        assert np.isclose(
            result.loc[
                index,
                "es_diversification_benefit",
            ],
            result.loc[
                index,
                "standalone_es",
            ]
            - result.loc[
                index,
                "incremental_es",
            ],
        )


# Standalone risk is calculated from each counterparty's
# own simulated loss distribution.
def test_standalone_risk_uses_counterparty_loss_distribution():

    portfolio = pd.DataFrame({
        "counterparty": [
            "Counterparty A",
            "Counterparty B",
        ],
        "exposure": [
            100.0,
            50.0,
        ],
    })

    counterparty_losses = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [5.0, 0.0],
        [5.0, 3.0],
        [10.0, 5.0],
        [20.0, 10.0],
    ])

    portfolio_losses = (
        counterparty_losses.sum(
            axis=1
        )
    )

    confidence_level = 0.80

    portfolio_var = (
        portfolio_model
        .value_at_risk(
            portfolio_losses,
            confidence_level,
        )
    )

    portfolio_es = (
        portfolio_model
        .expected_shortfall(
            portfolio_losses,
            confidence_level,
        )
    )

    result = (
        portfolio_model
        ._calculate_risk_contributions(
            portfolio=portfolio,
            portfolio_losses=portfolio_losses,
            counterparty_losses=counterparty_losses,
            confidence_level=confidence_level,
            portfolio_var=portfolio_var,
            portfolio_es=portfolio_es,
        )
    )

    for index in result.index:

        individual_losses = (
            counterparty_losses[
                :,
                index,
            ]
        )

        expected_standalone_var = (
            portfolio_model
            .value_at_risk(
                individual_losses,
                confidence_level,
            )
        )

        expected_standalone_es = (
            portfolio_model
            .expected_shortfall(
                individual_losses,
                confidence_level,
            )
        )

        assert np.isclose(
            result.loc[
                index,
                "standalone_var",
            ],
            expected_standalone_var,
        )

        assert np.isclose(
            result.loc[
                index,
                "standalone_es",
            ],
            expected_standalone_es,
        )


# Incremental risk equals full portfolio risk
# minus leave-one-out portfolio risk.
def test_incremental_risk_is_leave_one_out():

    portfolio = pd.DataFrame({
        "counterparty": [
            "Counterparty A",
            "Counterparty B",
        ],
        "exposure": [
            100.0,
            50.0,
        ],
    })

    counterparty_losses = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [5.0, 0.0],
        [5.0, 3.0],
        [10.0, 5.0],
        [20.0, 10.0],
    ])

    portfolio_losses = (
        counterparty_losses.sum(
            axis=1
        )
    )

    confidence_level = 0.80

    portfolio_var = (
        portfolio_model
        .value_at_risk(
            portfolio_losses,
            confidence_level,
        )
    )

    portfolio_es = (
        portfolio_model
        .expected_shortfall(
            portfolio_losses,
            confidence_level,
        )
    )

    result = (
        portfolio_model
        ._calculate_risk_contributions(
            portfolio=portfolio,
            portfolio_losses=portfolio_losses,
            counterparty_losses=counterparty_losses,
            confidence_level=confidence_level,
            portfolio_var=portfolio_var,
            portfolio_es=portfolio_es,
        )
    )

    for index in result.index:

        portfolio_without = (
            portfolio_losses
            - counterparty_losses[
                :,
                index,
            ]
        )

        var_without = (
            portfolio_model
            .value_at_risk(
                portfolio_without,
                confidence_level,
            )
        )

        es_without = (
            portfolio_model
            .expected_shortfall(
                portfolio_without,
                confidence_level,
            )
        )

        expected_incremental_var = (
            portfolio_var
            - var_without
        )

        expected_incremental_es = (
            portfolio_es
            - es_without
        )

        assert np.isclose(
            result.loc[
                index,
                "incremental_var",
            ],
            expected_incremental_var,
        )

        assert np.isclose(
            result.loc[
                index,
                "incremental_es",
            ],
            expected_incremental_es,
        )