#### Unit Tests for Credit Rating Migration ####

import numpy as np
import pandas as pd
import pytest

from quant_finance.credit.credit_migration import (
    get_rating_spreads,
    rating_to_spread,
    pd_to_equivalent_rating,
    get_adjusted_transition_probabilities,
    bond_value_with_credit_spread,
    migration_loss,
)


# Create a small synthetic migration matrix for tests.

def create_rating_migration_matrix():

    return pd.DataFrame({
        "current_rating": [
            "AAA",
            "AA",
            "A",
            "BBB",
            "BB",
            "B",
            "CCC",
            "D",
        ],
        "spread_decimal": [
            0.003,
            0.005,
            0.008,
            0.015,
            0.030,
            0.060,
            0.120,
            None,
        ],
        "AAA": [
            0.9150,
            0.0080,
            0.0010,
            0.0002,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        "AA": [
            0.0750,
            0.9000,
            0.0250,
            0.0030,
            0.0005,
            0.0001,
            0.0000,
            0.0000,
        ],
        "A": [
            0.0080,
            0.0750,
            0.8900,
            0.0350,
            0.0050,
            0.0010,
            0.0005,
            0.0000,
        ],
        "BBB": [
            0.0010,
            0.0120,
            0.0650,
            0.8500,
            0.0350,
            0.0050,
            0.0015,
            0.0000,
        ],
        "BB": [
            0.0003,
            0.0020,
            0.0120,
            0.0750,
            0.8000,
            0.0500,
            0.0080,
            0.0000,
        ],
        "B": [
            0.0001,
            0.0010,
            0.0030,
            0.0200,
            0.1000,
            0.7800,
            0.0500,
            0.0000,
        ],
        "CCC": [
            0.0001,
            0.0005,
            0.0010,
            0.0050,
            0.0250,
            0.0800,
            0.6000,
            0.0000,
        ],
        "D": [
            0.0005,
            0.0015,
            0.0030,
            0.0118,
            0.0345,
            0.0839,
            0.3400,
            1.0000,
        ],
    })


# 1. Rating spreads are extracted correctly.

def test_get_rating_spreads():

    matrix = create_rating_migration_matrix()

    spreads = get_rating_spreads(
        matrix
    )

    assert spreads["AAA"] == pytest.approx(
        0.003
    )

    assert spreads["BBB"] == pytest.approx(
        0.015
    )


# 2. Rating is mapped to the correct spread.

def test_rating_to_spread():

    matrix = create_rating_migration_matrix()

    spreads = get_rating_spreads(
        matrix
    )

    spread = rating_to_spread(
        rating="A",
        rating_spreads=spreads,
    )

    assert spread == pytest.approx(
        0.008
    )


# 3. PD is mapped to the closest equivalent rating.

def test_pd_to_equivalent_rating():

    matrix = create_rating_migration_matrix()

    rating = pd_to_equivalent_rating(
        pd=0.012,
        rating_migration_matrix=matrix,
    )

    assert rating == "BBB"


# 4. Adjusted transition probabilities sum to one.

def test_adjusted_transition_probabilities_sum_to_one():

    matrix = create_rating_migration_matrix()

    probabilities = (
        get_adjusted_transition_probabilities(
            current_rating="BBB",
            pd=0.02,
            rating_migration_matrix=matrix,
        )
    )

    assert probabilities.sum() == pytest.approx(
        1.0
    )


# 5. Adjusted transition probabilities preserve the counterparty PD.

def test_adjusted_transition_probabilities_preserve_pd():

    matrix = create_rating_migration_matrix()

    pd_value = 0.02

    probabilities = (
        get_adjusted_transition_probabilities(
            current_rating="BBB",
            pd=pd_value,
            rating_migration_matrix=matrix,
        )
    )

    assert probabilities[-1] == pytest.approx(
        pd_value
    )


# 6. Higher spread reduces bond value.

def test_higher_spread_reduces_bond_value():

    low_spread_value = bond_value_with_credit_spread(
        face_value=100,
        coupon_rate=0.04,
        maturity=5,
        payment_frequency=2,
        risk_free_rate=0.02,
        credit_spread=0.005,
    )

    high_spread_value = bond_value_with_credit_spread(
        face_value=100,
        coupon_rate=0.04,
        maturity=5,
        payment_frequency=2,
        risk_free_rate=0.02,
        credit_spread=0.030,
    )

    assert (
        high_spread_value
        < low_spread_value
    )


# 7. Downgrade produces a positive migration loss.

def test_downgrade_produces_loss():

    matrix = create_rating_migration_matrix()

    spreads = get_rating_spreads(
        matrix
    )

    loss = migration_loss(
        exposure=100,
        current_rating="A",
        new_rating="BBB",
        coupon_rate=0.04,
        maturity=5,
        payment_frequency=2,
        risk_free_rate=0.02,
        rating_spreads=spreads,
    )

    assert loss > 0


# 8. Upgrade produces a mark-to-market gain.

def test_upgrade_produces_gain():

    matrix = create_rating_migration_matrix()

    spreads = get_rating_spreads(
        matrix
    )

    loss = migration_loss(
        exposure=100,
        current_rating="A",
        new_rating="AA",
        coupon_rate=0.04,
        maturity=5,
        payment_frequency=2,
        risk_free_rate=0.02,
        rating_spreads=spreads,
    )

    assert loss < 0