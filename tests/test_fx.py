#### Unit Tests Foreign Exchange Rates ####

import pandas as pd
import pytest

import quant_finance.market_data.fx as fx


def create_fallback_market_data():

    return pd.DataFrame({
        "data_type": [
            "fx",
            "fx",
            "fx",
        ],
        "currency": [
            "CHF",
            "USD",
            "EUR",
        ],
        "base_currency": [
            "CHF",
            "CHF",
            "CHF",
        ],
        "value": [
            1.0,
            0.80,
            0.94,
        ],
    })


# 1. Base currency returns one.

def test_base_currency_fx():

    rate, source = fx.get_fx_rate(
        currency="USD",
        base_currency="USD",
        return_source=True,
    )

    assert rate == 1.0
    assert source == "base_currency"


# 2. Automatic FX retrieval works.

def test_automatic_fx_rate(monkeypatch):

    monkeypatch.setattr(
        fx,
        "_get_live_fx_rate",
        lambda currency, base_currency: 0.81,
    )

    rate, source = fx.get_fx_rate(
        currency="CHF",
        base_currency="USD",
        return_source=True,
    )

    assert rate == 0.81
    assert source == "automatic_market_data"


# 3. Direct fallback rate works with the snapshot reference currency.

def test_fx_fallback_to_snapshot_base(monkeypatch):

    market_data = create_fallback_market_data()

    def fail_live_fx(currency, base_currency):
        raise ValueError("Automatic retrieval failed.")

    monkeypatch.setattr(
        fx,
        "_get_live_fx_rate",
        fail_live_fx,
    )

    rate, source = fx.get_fx_rate(
        currency="USD",
        base_currency="CHF",
        market_data=market_data,
        return_source=True,
    )

    assert rate == pytest.approx(0.80)
    assert source == "fallback_snapshot"


# 4. Cross-rate fallback works for another model base currency.

def test_fx_fallback_cross_rate(monkeypatch):

    market_data = create_fallback_market_data()

    def fail_live_fx(currency, base_currency):
        raise ValueError("Automatic retrieval failed.")

    monkeypatch.setattr(
        fx,
        "_get_live_fx_rate",
        fail_live_fx,
    )

    rate, source = fx.get_fx_rate(
        currency="EUR",
        base_currency="USD",
        market_data=market_data,
        return_source=True,
    )

    expected_rate = 0.94 / 0.80

    assert rate == pytest.approx(expected_rate)
    assert source == "fallback_snapshot"


# 5. Missing currency in the fallback snapshot raises an error.

def test_missing_fx_fallback(monkeypatch):

    market_data = create_fallback_market_data()

    def fail_live_fx(currency, base_currency):
        raise ValueError("Automatic retrieval failed.")

    monkeypatch.setattr(
        fx,
        "_get_live_fx_rate",
        fail_live_fx,
    )

    with pytest.raises(ValueError):
        fx.get_fx_rate(
            currency="GBP",
            base_currency="USD",
            market_data=market_data,
        )