#### Unit Tests for Portfolio Credit Risk Analytics ####

import pytest

from quant_finance.credit.credit_risk_analytics import (
    incremental_risk,
    marginal_risk,
    diversification_benefit,
    diversification_ratio,
)


# 1. Incremental risk is calculated correctly.
def test_incremental_risk():

    result = incremental_risk(
        old_rc=10.0,
        new_rc=12.5,
    )

    assert result == pytest.approx(2.5)


# 2. Incremental risk can be negative.
def test_incremental_risk_negative():

    result = incremental_risk(
        old_rc=10.0,
        new_rc=9.0,
    )

    assert result == pytest.approx(-1.0)


# 3. Marginal risk is calculated correctly.
def test_marginal_risk():

    result = marginal_risk(
        old_rc=10.0,
        new_rc=12.0,
        added_asset=20.0,
    )

    assert result == pytest.approx(0.10)


# 4. Added asset must be positive.
def test_marginal_risk_invalid_added_asset():

    with pytest.raises(ValueError):
        marginal_risk(
            old_rc=10.0,
            new_rc=12.0,
            added_asset=0.0,
        )


# 5. Diversification benefit is calculated correctly.
def test_diversification_benefit():

    result = diversification_benefit(
        standalone_rc=5.0,
        incremental_rc=3.0,
    )

    assert result == pytest.approx(2.0)


# 6. Diversification ratio is calculated correctly.
def test_diversification_ratio():

    result = diversification_ratio(
        standalone_rc=5.0,
        incremental_rc=3.0,
    )

    assert result == pytest.approx(0.60)


# 7. Standalone risk charge must be positive.
def test_diversification_ratio_invalid_standalone_rc():

    with pytest.raises(ValueError):
        diversification_ratio(
            standalone_rc=0.0,
            incremental_rc=3.0,
        )
