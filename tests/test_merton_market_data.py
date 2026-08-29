#### Unit Tests Merton Market Data ####

import numpy as np
import pandas as pd

from quant_finance.credit.merton_market_data import (
    estimate_equity_volatility,
    estimate_equity_value,
    estimate_debt_default_point,
)


# 1. Equity volatility is calculated correctly.

def test_estimate_equity_volatility(monkeypatch):

    prices = pd.DataFrame(
        {
            "Adj Close": [
                100.0,
                105.0,
                103.0,
                108.0,
            ]
        }
    )

    monkeypatch.setattr(
        "quant_finance.credit.merton_market_data.yf.download",
        lambda *args, **kwargs: prices,
    )

    result = estimate_equity_volatility(
        ticker="TEST",
        start_date="2025-01-01",
        end_date="2026-01-01",
    )

    log_returns = np.log(
        prices["Adj Close"]
        / prices["Adj Close"].shift(1)
    ).dropna()

    expected = (
        log_returns.std(ddof=1)
        * np.sqrt(252)
    )

    assert np.isclose(
        result,
        expected,
    )


# 2. Equity market value is calculated correctly.

def test_estimate_equity_value(monkeypatch):

    class FakeTicker:

        def get_info(self):

            return {
                "sharesOutstanding": 1000
            }

        def history(
            self,
            period,
            auto_adjust,
        ):

            return pd.DataFrame(
                {
                    "Close": [
                        50.0,
                        55.0,
                    ]
                }
            )

    monkeypatch.setattr(
        "quant_finance.credit.merton_market_data.yf.Ticker",
        lambda ticker: FakeTicker(),
    )

    result = estimate_equity_value(
        ticker="TEST"
    )

    assert result == 55000.0


# 3. Debt default point is calculated correctly.

def test_estimate_debt_default_point(monkeypatch):

    class FakeTicker:

        def get_balance_sheet(
            self,
            freq,
        ):

            return pd.DataFrame(
                {
                    pd.Timestamp(
                        "2025-12-31"
                    ): [
                        20.0,
                        80.0,
                    ]
                },
                index=[
                    "CurrentDebt",
                    "LongTermDebt",
                ],
            )

    monkeypatch.setattr(
        "quant_finance.credit.merton_market_data.yf.Ticker",
        lambda ticker: FakeTicker(),
    )

    debt, short_term_debt, long_term_debt = (
        estimate_debt_default_point(
            ticker="TEST"
        )
    )

    assert short_term_debt == 20.0
    assert long_term_debt == 80.0
    assert debt == 60.0