#### Unit Tests for Portfolio Credit Risk Analytics ####

import numpy as np
import pytest

from quant_finance.credit.credit_risk_analytics import (
    expected_loss,
    portfolio_expected_loss,
    incremental_risk,
    marginal_risk,
    diversification_benefit,
    diversification_ratio,
    value_at_risk,
    expected_shortfall,
    unexpected_loss,
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


# 8. Expected loss for a single counterparty is calculated correctly.

def test_expected_loss():

    result = expected_loss(
        exposure=100,
        pd=0.02,
        lgd=0.40,
    )

    assert result == pytest.approx(0.8)


# 9. Portfolio expected loss is calculated correctly.

def test_portfolio_expected_loss():

    result = portfolio_expected_loss(
        exposures=[100, 50],
        pds=[0.02, 0.01],
        lgds=[0.40, 0.50],
    )

    expected = (
        100 * 0.02 * 0.40
        + 50 * 0.01 * 0.50
    )

    assert result == pytest.approx(
        expected
    )


# 10. PD must lie between zero and one.

def test_expected_loss_invalid_pd():

    with pytest.raises(ValueError):

        expected_loss(
            exposure=100,
            pd=1.20,
            lgd=0.40,
        )


# 11. Value at Risk returns the requested loss quantile.

def test_value_at_risk():

    losses = np.array([
        0,
        10,
        20,
        30,
        40,
    ])

    var = value_at_risk(
        losses=losses,
        confidence_level=0.80,
    )

    assert var == pytest.approx(
        32.0
    )


# 12. Expected Shortfall averages the worst tail scenarios.

def test_expected_shortfall():

    losses = np.array([
        0,
        10,
        20,
        30,
        40,
    ])

    es = expected_shortfall(
        losses=losses,
        confidence_level=0.80,
    )

    assert es == pytest.approx(
        40.0
    )


# 13. Unexpected loss equals VaR minus expected loss.

def test_unexpected_loss():

    result = unexpected_loss(
        value_at_risk_value=25,
        expected_loss_value=10,
    )

    assert result == pytest.approx(
        15.0
    )