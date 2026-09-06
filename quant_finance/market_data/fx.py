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


# Normalize an optional market-data date.
#
# None or "latest" means that the latest available observation
# should be used.
def _normalize_market_data_date(market_data_date):

    if market_data_date is None:
        return None

    if (
        isinstance(market_data_date, str)
        and market_data_date.strip().lower() == "latest"
    ):
        return None

    try:
        normalized_date = pd.Timestamp(
            market_data_date
        )

    except Exception as error:
        raise ValueError(
            "market_data_date must be 'latest' or a valid date."
        ) from error

    if pd.isna(normalized_date):
        raise ValueError(
            "market_data_date must be 'latest' or a valid date."
        )

    if normalized_date.tzinfo is not None:
        normalized_date = (
            normalized_date
            .tz_localize(None)
        )

    return normalized_date.normalize()


# Check that a fallback snapshot does not come after
# an explicitly requested market-data date.
def _validate_fallback_snapshot_date(
    data,
    market_data_date,
):

    requested_date = _normalize_market_data_date(
        market_data_date
    )

    if requested_date is None:
        return

    if "as_of_date" not in data.columns:
        raise ValueError(
            "Fallback market data must contain as_of_date "
            "when a specific market_data_date is requested."
        )

    snapshot_dates = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    )

    if snapshot_dates.isna().any():
        raise ValueError(
            "Fallback market data contains an invalid as_of_date."
        )

    snapshot_dates = (
        snapshot_dates
        .dt
        .normalize()
    )

    if snapshot_dates.nunique() != 1:
        raise ValueError(
            "Fallback market data must use one common as_of_date."
        )

    snapshot_date = snapshot_dates.iloc[0]

    if snapshot_date > requested_date:
        raise ValueError(
            "Fallback market-data snapshot is later than the "
            "requested market_data_date."
        )


# Retrieve ECB euro reference rates.
#
# If market_data_date is None, the latest published reference
# rates are used.
#
# If a specific date is supplied, the historical ECB time series
# is searched and the latest available observation on or before
# that date is selected. This automatically handles weekends and
# TARGET closing days.
def _get_ecb_reference_rates(
    market_data_date=None,
):

    requested_date = _normalize_market_data_date(
        market_data_date
    )

    # Latest available ECB reference rates.
    if requested_date is None:

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

    # Historical ECB reference rates.
    else:

        url = (
            "https://www.ecb.europa.eu/stats/eurofxref/"
            "eurofxref-hist.xml"
        )

        response = requests.get(
            url,
            timeout=20,
        )

        response.raise_for_status()

        root = ET.fromstring(
            response.content
        )

        selected_element = None
        selected_date = None

        for element in root.iter():

            time_value = element.attrib.get(
                "time"
            )

            if time_value is None:
                continue

            try:
                observation_date = pd.Timestamp(
                    time_value
                ).normalize()

            except Exception:
                continue

            if observation_date > requested_date:
                continue

            if (
                selected_date is None
                or observation_date > selected_date
            ):
                selected_date = observation_date
                selected_element = element

        if selected_element is None:
            raise ValueError(
                "No ECB FX reference-rate observation is available "
                "on or before the requested market_data_date."
            )

        rates = {
            "EUR": 1.0,
        }

        for element in selected_element:

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


# Calculate an automatic FX rate from local currency
# into the selected portfolio base currency.
def _get_live_fx_rate(
    currency,
    base_currency,
    market_data_date=None,
):

    if currency == base_currency:
        return 1.0

    rates = _get_ecb_reference_rates(
        market_data_date=market_data_date,
    )

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
    market_data_date=None,
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

    # If a historical date was requested, do not allow
    # a fallback snapshot from a later date.
    _validate_fallback_snapshot_date(
        data=fx_data,
        market_data_date=market_data_date,
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
#
# If market_data_date is specified, the latest available automatic
# observation on or before that date is used.
def get_fx_rate(
    currency,
    base_currency,
    market_data=None,
    return_source=False,
    market_data_date=None,
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

    requested_date = _normalize_market_data_date(
        market_data_date
    )

    # No conversion is necessary for the selected base currency.
    if currency == base_currency:

        rate = 1.0
        source = "base_currency"

    else:

        try:

            # Preserve the previous call signature when the
            # latest observation is requested.
            if requested_date is None:

                rate = _get_live_fx_rate(
                    currency=currency,
                    base_currency=base_currency,
                )

            else:

                rate = _get_live_fx_rate(
                    currency=currency,
                    base_currency=base_currency,
                    market_data_date=requested_date,
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
                    market_data_date=requested_date,
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