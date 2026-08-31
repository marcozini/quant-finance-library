#### Foreign Exchange Rates ####

import xml.etree.ElementTree as ET

import pandas as pd
import requests


# ECB reference rates are quoted as units of foreign currency
# per one euro.
#
# The Excel fallback snapshot stores all currencies relative
# to one common reference currency (currently CHF).
#
# Cross-rates are derived automatically, so the model base
# currency may be CHF, USD, EUR, or GBP without requiring
# separate fallback tables for every currency pair.


# Retrieve the latest ECB euro reference rates.

def _get_ecb_reference_rates():

    url = (
        "https://www.ecb.europa.eu/stats/eurofxref/"
        "eurofxref-daily.xml"
    )

    response = requests.get(
        url,
        timeout=15,
    )

    response.raise_for_status()

    root = ET.fromstring(
        response.content
    )

    rates = {
        "EUR": 1.0,
    }

    for element in root.iter():

        currency = element.attrib.get(
            "currency"
        )

        rate = element.attrib.get(
            "rate"
        )

        if (
            currency is not None
            and rate is not None
        ):
            rates[
                currency.upper()
            ] = float(rate)

    if len(rates) <= 1:
        raise ValueError(
            "No ECB foreign-exchange reference rates available."
        )

    return rates


# Calculate a live FX rate from local currency
# into the selected portfolio base currency.

def _get_live_fx_rate(
    currency,
    base_currency,
):

    if currency == base_currency:
        return 1.0

    rates = _get_ecb_reference_rates()

    if currency not in rates:
        raise ValueError(
            f"No ECB FX rate available for currency "
            f"'{currency}'."
        )

    if base_currency not in rates:
        raise ValueError(
            f"No ECB FX rate available for base currency "
            f"'{base_currency}'."
        )

    # ECB rates are foreign currency units per EUR.
    # Therefore:
    #
    # local -> base
    # = base units / EUR
    #   divided by local units / EUR

    fx_rate = (
        rates[base_currency]
        / rates[currency]
    )

    if fx_rate <= 0:
        raise ValueError(
            "Calculated FX rate must be strictly positive."
        )

    return float(fx_rate)


# Retrieve an FX fallback cross-rate from the frozen snapshot.

def _get_fallback_fx_rate(
    currency,
    base_currency,
    market_data,
):

    if market_data is None:
        raise ValueError(
            "No fallback market data provided."
        )

    required_columns = [
        "data_type",
        "currency",
        "base_currency",
        "value",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in market_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing fallback market-data columns: "
            f"{missing_columns}"
        )

    data = market_data.copy()

    data["data_type_normalized"] = (
        data["data_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["currency_normalized"] = (
        data["currency"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data["base_currency_normalized"] = (
        data["base_currency"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data["value_numeric"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    fx_data = data[
        data["data_type_normalized"] == "fx"
    ].copy()

    if fx_data.empty:
        raise ValueError(
            "No FX fallback data available."
        )

    # All stored fallback FX rates must share
    # the same reference currency.

    snapshot_bases = (
        fx_data[
            "base_currency_normalized"
        ]
        .dropna()
        .unique()
    )

    if len(snapshot_bases) != 1:
        raise ValueError(
            "FX fallback data must use one common "
            "snapshot reference currency."
        )

    snapshot_base = snapshot_bases[0]

    # Retrieve local currency -> snapshot reference.

    currency_rows = fx_data[
        fx_data["currency_normalized"]
        == currency
    ]

    if len(currency_rows) != 1:
        raise ValueError(
            f"Exactly one FX fallback rate must exist "
            f"for currency '{currency}'."
        )

    currency_to_snapshot = currency_rows[
        "value_numeric"
    ].iloc[0]

    # Retrieve selected model base currency -> snapshot reference.

    base_rows = fx_data[
        fx_data["currency_normalized"]
        == base_currency
    ]

    if len(base_rows) != 1:
        raise ValueError(
            f"Exactly one FX fallback rate must exist "
            f"for model base currency '{base_currency}'."
        )

    base_to_snapshot = base_rows[
        "value_numeric"
    ].iloc[0]

    if (
        pd.isna(currency_to_snapshot)
        or currency_to_snapshot <= 0
    ):
        raise ValueError(
            f"Invalid FX fallback rate for "
            f"currency '{currency}'."
        )

    if (
        pd.isna(base_to_snapshot)
        or base_to_snapshot <= 0
    ):
        raise ValueError(
            f"Invalid FX fallback rate for "
            f"base currency '{base_currency}'."
        )

    # Example with CHF as snapshot reference:
    #
    # EUR -> USD
    # = (EUR -> CHF) / (USD -> CHF)

    fx_rate = (
        currency_to_snapshot
        / base_to_snapshot
    )

    if fx_rate <= 0:
        raise ValueError(
            "Calculated fallback FX cross-rate "
            "must be strictly positive."
        )

    return float(fx_rate)


# Retrieve FX rate from local currency into portfolio base currency.
#
# Automatic ECB market data is attempted first.
# The frozen Excel snapshot is used only if automatic retrieval fails.

def get_fx_rate(
    currency,
    base_currency,
    market_data=None,
    return_source=False,
):

    currency = str(
        currency
    ).upper().strip()

    base_currency = str(
        base_currency
    ).upper().strip()

    if not currency:
        raise ValueError(
            "Currency must not be empty."
        )

    if not base_currency:
        raise ValueError(
            "Base currency must not be empty."
        )

    # No conversion is necessary for the selected base currency.
    if currency == base_currency:

        rate = 1.0
        source = "base_currency"

    else:

        try:

            rate = _get_live_fx_rate(
                currency=currency,
                base_currency=base_currency,
            )

            source = "automatic_market_data"

        except Exception as live_error:

            if market_data is None:
                raise live_error

            try:

                rate = _get_fallback_fx_rate(
                    currency=currency,
                    base_currency=base_currency,
                    market_data=market_data,
                )

                source = "fallback_snapshot"

            except Exception as fallback_error:

                raise ValueError(
                    f"Could not obtain FX rate for "
                    f"'{currency}' to '{base_currency}'. "
                    f"Automatic retrieval failed with: "
                    f"{live_error}. "
                    f"Fallback retrieval failed with: "
                    f"{fallback_error}."
                ) from fallback_error

    if return_source:
        return rate, source

    return rate