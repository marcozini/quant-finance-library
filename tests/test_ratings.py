#### Unit Tests for Credit Ratings ####

import pytest

from quant_finance.credit.ratings import TRANSITION_MATRIX, rating_to_pd


# 1. Correct probability of default is returned
def test_rating_to_pd():

    pd = rating_to_pd("BBB")

    assert pd == pytest.approx(0.0040)


# 2. Lowercase rating input is handled correctly
def test_rating_to_pd_lowercase():

    pd = rating_to_pd("bbb")

    assert pd == pytest.approx(0.0040)


# 3. Default rating returns probability of default equal to 1
def test_default_rating():

    pd = rating_to_pd("D")

    assert pd == pytest.approx(1.0)


# 4. Invalid rating raises an error
def test_invalid_rating():

    with pytest.raises(ValueError):
        rating_to_pd("XYZ")


# 5. Every row of the transition matrix sums to 1
def test_transition_matrix_rows_sum_to_one():

    for rating, transitions in TRANSITION_MATRIX.items():

        row_sum = sum(transitions.values())

        assert row_sum == pytest.approx(1.0)


# 6. All transition probabilities are between 0 and 1
def test_transition_probabilities_are_valid():

    for transitions in TRANSITION_MATRIX.values():

        for probability in transitions.values():

            assert 0.0 <= probability <= 1.0


# 7. Default state is absorbing
def test_default_state_is_absorbing():

    assert TRANSITION_MATRIX["D"]["D"] == pytest.approx(1.0)

    for rating, probability in TRANSITION_MATRIX["D"].items():

        if rating != "D":
            assert probability == pytest.approx(0.0)


# 8. All rows and columns use the same rating states
def test_transition_matrix_states_are_consistent():

    rating_states = set(TRANSITION_MATRIX.keys())

    for transitions in TRANSITION_MATRIX.values():

        assert set(transitions.keys()) == rating_states