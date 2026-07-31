#### Unit Tests for Merton Model ####

import pytest

from quant_finance.credit.merton import (
    merton_equity_value,
    merton_debt_value,
    distance_to_default,
    merton_probability_of_default,
)


# 1. Equity value matches a known benchmark
def test_merton_equity_value_known_value():

    equity_value = merton_equity_value(
        asset_value=100,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    assert equity_value == pytest.approx(24.588835, rel=1e-6)


# 2. Equity plus risky debt equals total asset value
def test_equity_plus_debt_equals_asset_value():

    asset_value = 100

    equity_value = merton_equity_value(
        asset_value=asset_value,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    debt_value = merton_debt_value(
        asset_value=asset_value,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    assert equity_value + debt_value == pytest.approx(asset_value)


# 3. Distance to default matches a known benchmark
def test_distance_to_default_known_value():

    dd = distance_to_default(
        asset_value=100,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    assert dd == pytest.approx(1.265718, rel=1e-6)


# 4. Probability of default matches a known benchmark
def test_probability_of_default_known_value():

    pd = merton_probability_of_default(
        asset_value=100,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    assert pd == pytest.approx(0.1028, abs=1e-4)
    assert 0 <= pd <= 1


# 5. Higher asset value produces a lower PD
def test_higher_asset_value_reduces_pd():

    lower_asset_pd = merton_probability_of_default(
        asset_value=100,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    higher_asset_pd = merton_probability_of_default(
        asset_value=120,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    assert higher_asset_pd < lower_asset_pd


# 6. Higher debt produces a higher PD
def test_higher_debt_increases_pd():

    lower_debt_pd = merton_probability_of_default(
        asset_value=100,
        debt_face_value=80,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    higher_debt_pd = merton_probability_of_default(
        asset_value=100,
        debt_face_value=100,
        risk_free_rate=0.05,
        asset_volatility=0.20,
        time_to_maturity=1.0,
    )

    assert higher_debt_pd > lower_debt_pd