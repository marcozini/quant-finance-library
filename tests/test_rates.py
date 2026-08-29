#### Unit Tests Risk-Free Rates ####

import pytest

import quant_finance.market_data.rates as rates


# 1. Currency routing works correctly.

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


# 2. Unsupported currency raises an error.

def test_invalid_currency():

    with pytest.raises(ValueError):

        rates.get_risk_free_rate(
            currency="JPY"
        )