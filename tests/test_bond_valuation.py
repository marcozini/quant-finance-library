#### Unit Tests for Bond Valuation ####

import numpy as np
import pytest

from quant_finance.credit.bond_valuation import (
    bond_cash_flows,
    bond_price,
    implied_spread,
)


# 1. Annual coupon cash flows are generated correctly.
def test_bond_cash_flows_annual():

    times, cash_flows = bond_cash_flows(
        nominal=100,
        coupon_rate=0.05,
        maturity=3,
        frequency=1,
    )

    np.testing.assert_allclose(times, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cash_flows, [5.0, 5.0, 105.0])


# 2. Semiannual coupon cash flows are generated correctly.
def test_bond_cash_flows_semiannual():

    times, cash_flows = bond_cash_flows(
        nominal=100,
        coupon_rate=0.06,
        maturity=2,
        frequency=2,
    )

    np.testing.assert_allclose(times, [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(cash_flows, [3.0, 3.0, 3.0, 103.0])


# 3. Zero-coupon bond cash flows are generated correctly.
def test_zero_coupon_bond_cash_flows():

    times, cash_flows = bond_cash_flows(
        nominal=100,
        coupon_rate=0.0,
        maturity=2,
        frequency=1,
    )

    np.testing.assert_allclose(times, [1.0, 2.0])
    np.testing.assert_allclose(cash_flows, [0.0, 100.0])


# 4. Bond price matches direct present-value calculation.
def test_bond_price():

    times = np.array([1.0, 2.0, 3.0])
    cash_flows = np.array([5.0, 5.0, 105.0])

    risk_free_rate = 0.02
    spread = 0.01

    expected_price = np.sum(
        cash_flows * np.exp(-(risk_free_rate + spread) * times)
    )

    price = bond_price(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rate,
        spread=spread,
    )

    assert price == pytest.approx(expected_price)


# 5. Bond pricing works with a risk-free term structure.
def test_bond_price_term_structure():

    times = np.array([1.0, 2.0, 3.0])
    cash_flows = np.array([5.0, 5.0, 105.0])
    risk_free_rates = np.array([0.01, 0.015, 0.02])
    spread = 0.01

    expected_price = np.sum(
        cash_flows * np.exp(-(risk_free_rates + spread) * times)
    )

    price = bond_price(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        spread=spread,
    )

    assert price == pytest.approx(expected_price)


# 6. Implied spread recovers the spread used to generate the market value.
def test_implied_spread():

    times, cash_flows = bond_cash_flows(
        nominal=100,
        coupon_rate=0.05,
        maturity=5,
        frequency=1,
    )

    risk_free_rates = 0.02
    true_spread = 0.025

    market_value = bond_price(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        spread=true_spread,
    )

    calculated_spread = implied_spread(
        market_value=market_value,
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
    )

    assert calculated_spread == pytest.approx(true_spread, abs=1e-7)


# 7. Invalid bond inputs raise an error.
def test_bond_cash_flows_invalid_input():

    with pytest.raises(ValueError):
        bond_cash_flows(
            nominal=-100,
            coupon_rate=0.05,
            maturity=3,
        )


# 8. Implied spread requires a valid bisection interval.
def test_implied_spread_invalid_interval():

    times = np.array([1.0, 2.0])
    cash_flows = np.array([5.0, 105.0])

    with pytest.raises(ValueError):
        implied_spread(
            market_value=100,
            times=times,
            cash_flows=cash_flows,
            risk_free_rates=0.02,
            lower=0.10,
            upper=0.05,
        )