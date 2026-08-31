#### Unit Tests Risk-Free Rates ####

import pandas as pd
import pytest

import quant_finance.market_data.rates as rates


def create_fallback_market_data():

    return pd.DataFrame({
        "data_type": [
            "risk_free",
            "risk_free",
        ],
        "currency": [
            "USD",
            "EUR",
        ],
        "tenor_years": [
            1.0,
            1.0,
        ],
        "value": [
            0.03625,
            0.0200,
        ],
    })


# 1. Automatic currency routing works correctly.

def test_get_risk_free_rate(monkeypatch):

    monkeypatch.setattr(
        rates,
        "_get_usd_risk_free_rate",
        lambda: 0.04,
    )

    result = rates.get_risk_free_rate(
        currency="usd"
    )

    assert result == 0.04


# 2. Unsupported currency without fallback raises an error.

def test_invalid_currency():

    with pytest.raises(ValueError):
        rates.get_risk_free_rate(
            currency="JPY"
        )


# 3. Fallback rate is used when automatic retrieval fails.

def test_risk_free_rate_fallback(monkeypatch):

    market_data = create_fallback_market_data()

    def fail_live_rate(currency):
        raise ValueError("Automatic retrieval failed.")

    monkeypatch.setattr(
        rates,
        "_get_live_risk_free_rate",
        fail_live_rate,
    )

    result = rates.get_risk_free_rate(
        currency="USD",
        market_data=market_data,
    )

    assert result == 0.03625


# 4. Rate source identifies automatic market data.

def test_automatic_rate_source(monkeypatch):

    monkeypatch.setattr(
        rates,
        "_get_live_risk_free_rate",
        lambda currency: 0.04,
    )

    rate, source = rates.get_risk_free_rate(
        currency="USD",
        return_source=True,
    )

    assert rate == 0.04
    assert source == "automatic_market_data"


# 5. Rate source identifies fallback snapshot.

def test_fallback_rate_source(monkeypatch):

    market_data = create_fallback_market_data()

    def fail_live_rate(currency):
        raise ValueError("Automatic retrieval failed.")

    monkeypatch.setattr(
        rates,
        "_get_live_risk_free_rate",
        fail_live_rate,
    )

    rate, source = rates.get_risk_free_rate(
        currency="USD",
        market_data=market_data,
        return_source=True,
    )

    assert rate == 0.03625
    assert source == "fallback_snapshot"


# 6. Missing fallback rate raises an error.

def test_missing_fallback_rate(monkeypatch):

    market_data = create_fallback_market_data()

    def fail_live_rate(currency):
        raise ValueError("Automatic retrieval failed.")

    monkeypatch.setattr(
        rates,
        "_get_live_risk_free_rate",
        fail_live_rate,
    )

    with pytest.raises(ValueError):
        rates.get_risk_free_rate(
            currency="GBP",
            market_data=market_data,
        )