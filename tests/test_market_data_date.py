#### Unit Tests Market Data Date ####

import numpy as np
import pandas as pd
import pytest

import quant_finance.market_data.fx as fx
import quant_finance.market_data.rates as rates
import quant_finance.credit.portfolio_model as portfolio_model


# =========================================================
# FX Date Selection
# =========================================================


# Historical ECB FX data should use the latest available
# observation on or before the requested date.
#
# The requested date below is Sunday, 30 August 2026.
# The latest available observation should therefore be
# Friday, 28 August 2026.
def test_historical_fx_uses_previous_available_date(
    monkeypatch,
):

    historical_xml = b"""
    <Envelope>
        <Cube>
            <Cube time="2026-08-28">
                <Cube currency="USD" rate="1.1700"/>
                <Cube currency="CHF" rate="0.9200"/>
            </Cube>
            <Cube time="2026-08-31">
                <Cube currency="USD" rate="1.1800"/>
                <Cube currency="CHF" rate="0.9250"/>
            </Cube>
        </Cube>
    </Envelope>
    """

    class MockResponse:

        content = historical_xml

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        fx.requests,
        "get",
        lambda *args, **kwargs: MockResponse(),
    )

    reference_rates = (
        fx._get_ecb_reference_rates(
            market_data_date="2026-08-30",
        )
    )

    assert (
        reference_rates["EUR"]
        == 1.0
    )

    assert (
        reference_rates["USD"]
        == 1.1700
    )

    assert (
        reference_rates["CHF"]
        == 0.9200
    )


# Explicit FX market-data date should be forwarded
# to the automatic market-data function.
def test_fx_market_data_date_is_forwarded(
    monkeypatch,
):

    captured = {}

    def mock_live_fx_rate(
        currency,
        base_currency,
        market_data_date=None,
    ):

        captured[
            "currency"
        ] = currency

        captured[
            "base_currency"
        ] = base_currency

        captured[
            "market_data_date"
        ] = market_data_date

        return 1.25

    monkeypatch.setattr(
        fx,
        "_get_live_fx_rate",
        mock_live_fx_rate,
    )

    result = fx.get_fx_rate(
        currency="EUR",
        base_currency="USD",
        market_data_date="2026-08-31",
    )

    assert (
        result
        == 1.25
    )

    assert (
        captured[
            "currency"
        ]
        == "EUR"
    )

    assert (
        captured[
            "base_currency"
        ]
        == "USD"
    )

    assert (
        captured[
            "market_data_date"
        ]
        == pd.Timestamp(
            "2026-08-31"
        )
    )


# A fallback snapshot must not contain information
# from after the requested historical market-data date.
def test_fx_rejects_future_fallback_snapshot(
    monkeypatch,
):

    def fail_live_fx_rate(
        *args,
        **kwargs,
    ):

        raise ValueError(
            "Automatic FX retrieval unavailable."
        )

    monkeypatch.setattr(
        fx,
        "_get_live_fx_rate",
        fail_live_fx_rate,
    )

    market_data = pd.DataFrame({

        "data_type": [
            "fx",
            "fx",
        ],

        "currency": [
            "EUR",
            "USD",
        ],

        "base_currency": [
            "CHF",
            "CHF",
        ],

        "value": [
            0.94,
            0.80,
        ],

        "as_of_date": [
            "2026-09-01",
            "2026-09-01",
        ],
    })

    with pytest.raises(
        ValueError,
        match=(
            "Fallback market-data snapshot is later than"
        ),
    ):

        fx.get_fx_rate(
            currency="EUR",
            base_currency="USD",
            market_data=market_data,
            market_data_date="2026-08-31",
        )


# A fallback snapshot from before the requested date
# remains acceptable.
def test_fx_accepts_earlier_fallback_snapshot(
    monkeypatch,
):

    def fail_live_fx_rate(
        *args,
        **kwargs,
    ):

        raise ValueError(
            "Automatic FX retrieval unavailable."
        )

    monkeypatch.setattr(
        fx,
        "_get_live_fx_rate",
        fail_live_fx_rate,
    )

    market_data = pd.DataFrame({

        "data_type": [
            "fx",
            "fx",
        ],

        "currency": [
            "EUR",
            "USD",
        ],

        "base_currency": [
            "CHF",
            "CHF",
        ],

        "value": [
            0.94,
            0.80,
        ],

        "as_of_date": [
            "2026-08-28",
            "2026-08-28",
        ],
    })

    result, source = fx.get_fx_rate(
        currency="EUR",
        base_currency="USD",
        market_data=market_data,
        market_data_date="2026-08-31",
        return_source=True,
    )

    expected_rate = (
        0.94
        / 0.80
    )

    assert result == pytest.approx(
        expected_rate
    )

    assert (
        source
        == "fallback_snapshot"
    )


# =========================================================
# Risk-Free Rate Date Selection
# =========================================================


# FRED-style data should select the latest available
# observation on or before the requested date.
#
# 30 August 2026 is a Sunday, so the observation from
# 28 August should be selected rather than 31 August.
def test_fred_rate_uses_previous_available_date():

    rate_data = pd.DataFrame({

        "DATE": [
            "2026-08-27",
            "2026-08-28",
            "2026-08-31",
        ],

        "DGS1": [
            3.70,
            3.75,
            3.80,
        ],
    })

    result = rates._select_fred_observation(
        rate_data=rate_data,
        value_column="DGS1",
        market_data_date="2026-08-30",
        empty_error_message=(
            "No rate available."
        ),
    )

    assert (
        result
        == 3.75
    )


# Explicit risk-free market-data date should be forwarded
# to the currency-specific automatic rate function.
def test_risk_free_market_data_date_is_forwarded(
    monkeypatch,
):

    captured = {}

    def mock_live_rate(
        currency,
        market_data_date=None,
    ):

        captured[
            "currency"
        ] = currency

        captured[
            "market_data_date"
        ] = market_data_date

        return 0.04

    monkeypatch.setattr(
        rates,
        "_get_live_risk_free_rate",
        mock_live_rate,
    )

    result = rates.get_risk_free_rate(
        currency="USD",
        market_data_date="2026-08-31",
    )

    assert (
        result
        == 0.04
    )

    assert (
        captured[
            "currency"
        ]
        == "USD"
    )

    assert (
        captured[
            "market_data_date"
        ]
        == pd.Timestamp(
            "2026-08-31"
        )
    )


# A future fallback risk-free snapshot must not be used
# for an earlier requested market-data date.
def test_risk_free_rejects_future_fallback_snapshot(
    monkeypatch,
):

    def fail_live_rate(
        *args,
        **kwargs,
    ):

        raise ValueError(
            "Automatic rate retrieval unavailable."
        )

    monkeypatch.setattr(
        rates,
        "_get_live_risk_free_rate",
        fail_live_rate,
    )

    market_data = pd.DataFrame({

        "data_type": [
            "risk_free",
        ],

        "currency": [
            "USD",
        ],

        "tenor_years": [
            1.0,
        ],

        "value": [
            0.04,
        ],

        "as_of_date": [
            "2026-09-01",
        ],
    })

    with pytest.raises(
        ValueError,
        match=(
            "Fallback market-data snapshot is later than"
        ),
    ):

        rates.get_risk_free_rate(
            currency="USD",
            market_data=market_data,
            tenor_years=1.0,
            market_data_date="2026-08-31",
        )


# =========================================================
# Portfolio Model Integration
# =========================================================


# A specific market_data_date from the model settings
# must reach both the risk-free-rate and FX layers.
def test_portfolio_model_forwards_market_data_date(
    monkeypatch,
):

    portfolio = pd.DataFrame({

        "counterparty": [
            "Test Co",
        ],

        "exposure": [
            100.0,
        ],

        "lgd": [
            0.40,
        ],

        "pd": [
            0.01,
        ],

        "expected_loss": [
            0.40,
        ],
    })

    settings = {

        "confidence_level":
            0.75,

        "factor_lookback_years":
            3,

        "base_currency":
            "USD",

        "market_data_date":
            "2026-08-31",
    }

    factor_proxies = pd.DataFrame()

    rating_migration_matrix = (
        pd.DataFrame()
    )

    market_data = pd.DataFrame()

    portfolio_losses = np.array([
        0.0,
        0.0,
        5.0,
        10.0,
    ])

    counterparty_losses = (
        portfolio_losses
        .reshape(
            -1,
            1,
        )
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

    captured = {}


    # -----------------------------
    # Excel loading
    # -----------------------------

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


    # -----------------------------
    # Validation
    # -----------------------------

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


    # -----------------------------
    # Capture market-data date
    # -----------------------------

    def mock_complete_rates(
        portfolio,
        market_data,
        market_data_date=None,
    ):

        captured[
            "risk_free_date"
        ] = market_data_date

        return portfolio


    def mock_convert_fx(
        portfolio,
        base_currency,
        market_data,
        market_data_date=None,
    ):

        captured[
            "fx_date"
        ] = market_data_date

        return portfolio


    monkeypatch.setattr(
        portfolio_model,
        "_complete_risk_free_rates",
        mock_complete_rates,
    )

    monkeypatch.setattr(
        portfolio_model,
        "_convert_exposures_to_base_currency",
        mock_convert_fx,
    )


    # -----------------------------
    # Other model components
    # -----------------------------

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


    # -----------------------------
    # Run model
    # -----------------------------

    results = (
        portfolio_model
        .run_credit_portfolio_model(
            "dummy.xlsx"
        )
    )


    # -----------------------------
    # Assertions
    # -----------------------------

    expected_date = pd.Timestamp(
        "2026-08-31"
    ).date()

    assert (
        captured[
            "risk_free_date"
        ]
        == expected_date
    )

    assert (
        captured[
            "fx_date"
        ]
        == expected_date
    )

    assert (
        results[
            "summary"
        ][
            "market_data_date"
        ]
        == "2026-08-31"
    )