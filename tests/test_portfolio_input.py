#### Unit Tests for Credit Portfolio Input ####

import pandas as pd
import pytest

from quant_finance.credit.portfolio_input import validate_portfolio


# Create a small valid portfolio for validation tests.

def create_valid_portfolio():

    return pd.DataFrame({
        "counterparty": ["Company A", "Company B"],
        "ticker": [None, "AAPL"],
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