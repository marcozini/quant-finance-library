#### Unit Tests for Credit Risk Migration ####

import pytest

from quant_finance.credit.bond_valuation import bond_price
from quant_finance.credit.credit_risk_migration import (
    rating_to_spread,
    migrated_bond_value,
    migration_loss,
)


# 1. Rating spread is returned correctly.
def test_rating_to_spread():

    assert rating_to_spread("BBB") == pytest.approx(0.015)


# 2. Lowercase ratings are accepted.
def test_rating_to_spread_lowercase():

    assert rating_to_spread("aa") == pytest.approx(0.005)


# 3. Invalid ratings raise an error.
def test_rating_to_spread_invalid_rating():

    with pytest.raises(ValueError):
        rating_to_spread("XYZ")


# 4. Bond is revalued using the migrated rating spread.
def test_migrated_bond_value():

    times = [1.0, 2.0]
    cash_flows = [5.0, 105.0]
    risk_free_rates = 0.02

    expected_value = bond_price(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        spread=0.015,
    )

    value = migrated_bond_value(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        migrated_rating="BBB",
    )

    assert value == pytest.approx(expected_value)


# 5. Downgrade produces a positive migration loss.
def test_migration_loss_downgrade():

    times = [1.0, 2.0]
    cash_flows = [5.0, 105.0]
    risk_free_rates = 0.02

    current_value = bond_price(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        spread=0.008,
    )

    migrated_value, loss = migration_loss(
        current_value=current_value,
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        migrated_rating="BBB",
    )

    assert migrated_value < current_value
    assert loss > 0


# 6. Upgrade produces a negative migration loss.
def test_migration_loss_upgrade():

    times = [1.0, 2.0]
    cash_flows = [5.0, 105.0]
    risk_free_rates = 0.02

    current_value = bond_price(
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        spread=0.015,
    )

    migrated_value, loss = migration_loss(
        current_value=current_value,
        times=times,
        cash_flows=cash_flows,
        risk_free_rates=risk_free_rates,
        migrated_rating="A",
    )

    assert migrated_value > current_value
    assert loss < 0