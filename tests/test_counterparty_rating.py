#### Unit Tests for Counterparty Rating Logic ####

import pytest

from quant_finance.credit.counterparty_rating import get_counterparty_pd


# 1. Rating-based PD is returned correctly
def test_get_counterparty_pd_rating():

    pd = get_counterparty_pd(
        method="rating",
        rating="BBB",
    )

    assert pd == pytest.approx(0.0040)


# 2. Merton-based PD is returned correctly
def test_get_counterparty_pd_merton():

    pd = get_counterparty_pd(
        method="merton",
        merton_pd=0.012,
    )

    assert pd == pytest.approx(0.012)


# 3. Missing rating raises an error
def test_missing_rating():

    with pytest.raises(ValueError):
        get_counterparty_pd(method="rating")


# 4. Missing Merton PD raises an error
def test_missing_merton_pd():

    with pytest.raises(ValueError):
        get_counterparty_pd(method="merton")


# 5. Invalid Merton PD raises an error
def test_invalid_merton_pd():

    with pytest.raises(ValueError):
        get_counterparty_pd(
            method="merton",
            merton_pd=1.5,
        )


# 6. Invalid method raises an error
def test_invalid_method():

    with pytest.raises(ValueError):
        get_counterparty_pd(method="unknown")