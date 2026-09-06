#### Risk-Free Rates ####

import unicodedata
from datetime import timedelta
from io import BytesIO, StringIO

import pandas as pd
import requests


# Risk-free-rate proxies used in this module:
#
# USD: 1-year US Treasury yield.
# EUR: 1-year euro area AAA zero-coupon spot rate.
# GBP: SONIA overnight rate used as a short-term proxy.
# CHF: approximately 12-month Geldmarktbuchforderung
#      der Schweizerischen Eidgenossenschaft (GMBF).
#
# Notes:
# - GBP currently uses an overnight proxy rather than a true
#   one-year zero-coupon rate.
# - Stored flat policy-rate proxies in the Excel market-data
#   snapshot are simplified fallback inputs only. They are not
#   interpreted as calibrated market yield curves.


# Normalize an optional market-data date.
#
# None or "latest" means that the latest available observation
# should be used.
def _normalize_market_data_date(
    market_data_date,
):

    if market_data_date is None:
        return None

    if (
        isinstance(
            market_data_date,
            str,
        )
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

    if pd.isna(
        normalized_date
    ):
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


# Select the latest valid FRED observation on or before
# an optional market-data date.
def _select_fred_observation(
    rate_data,
    value_column,
    market_data_date,
    empty_error_message,
):

    if value_column not in rate_data.columns:

        raise ValueError(
            f"Expected FRED column '{value_column}' was not found."
        )

    data = rate_data.copy()

    date_column = None

    for column in data.columns:

        normalized_column = (
            str(column)
            .strip()
            .lower()
        )

        if normalized_column in {
            "date",
            "observation_date",
        }:

            date_column = column
            break

    # FRED CSV files normally place the date column first.
    if date_column is None:

        candidate_columns = [
            column
            for column in data.columns
            if column != value_column
        ]

        if not candidate_columns:

            raise ValueError(
                "No date column found in FRED data."
            )

        date_column = candidate_columns[0]

    data[
        "observation_date"
    ] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    data[
        "rate_value"
    ] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "observation_date",
            "rate_value",
        ]
    )

    requested_date = _normalize_market_data_date(
        market_data_date
    )

    if requested_date is not None:

        data = data[
            data["observation_date"]
            <= requested_date
        ]

    if data.empty:

        raise ValueError(
            empty_error_message
        )

    data = data.sort_values(
        "observation_date"
    )

    return float(
        data[
            "rate_value"
        ].iloc[-1]
    )


# Retrieve the latest available one-year US Treasury yield
# on or before an optional market-data date.
def _get_usd_risk_free_rate(
    market_data_date=None,
):

    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id=DGS1"
    )

    rate_data = pd.read_csv(
        url
    )

    selected_yield = _select_fred_observation(
        rate_data=rate_data,
        value_column="DGS1",
        market_data_date=market_data_date,
        empty_error_message=(
            "No one-year US Treasury yield data available "
            "on or before the requested date."
        ),
    )

    return float(
        selected_yield
        / 100
    )


# Retrieve the latest available one-year euro area AAA
# spot rate on or before an optional market-data date.
def _get_eur_risk_free_rate(
    market_data_date=None,
):

    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y"
    )

    requested_date = _normalize_market_data_date(
        market_data_date
    )

    params = {
        "format": "csvdata",
    }

    # For the latest observation, preserve the efficient
    # one-observation query.
    if requested_date is None:

        params[
            "lastNObservations"
        ] = 1

    # For a historical request, retrieve a short window
    # ending at the requested date and select the latest
    # available observation.
    else:

        start_date = (
            requested_date
            - timedelta(days=31)
        )

        params[
            "startPeriod"
        ] = start_date.date().isoformat()

        params[
            "endPeriod"
        ] = requested_date.date().isoformat()

    response = requests.get(
        url,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    rate_data = pd.read_csv(
        StringIO(
            response.text
        )
    )

    if rate_data.empty:

        raise ValueError(
            "No one-year EUR spot-rate data available."
        )

    if "OBS_VALUE" not in rate_data.columns:

        raise ValueError(
            "ECB EUR rate response does not contain OBS_VALUE."
        )

    rate_data[
        "rate_value"
    ] = pd.to_numeric(
        rate_data[
            "OBS_VALUE"
        ],
        errors="coerce",
    )

    if "TIME_PERIOD" in rate_data.columns:

        rate_data[
            "observation_date"
        ] = pd.to_datetime(
            rate_data[
                "TIME_PERIOD"
            ],
            errors="coerce",
        )

        rate_data = rate_data.dropna(
            subset=[
                "observation_date",
                "rate_value",
            ]
        )

        if requested_date is not None:

            rate_data = rate_data[
                rate_data[
                    "observation_date"
                ]
                <= requested_date
            ]

        rate_data = rate_data.sort_values(
            "observation_date"
        )

    else:

        rate_data = rate_data.dropna(
            subset=[
                "rate_value",
            ]
        )

    if rate_data.empty:

        raise ValueError(
            "No valid one-year EUR spot rate available "
            "on or before the requested date."
        )

    latest_yield = rate_data[
        "rate_value"
    ].iloc[-1]

    return float(
        latest_yield
        / 100
    )


# Retrieve GBP short-term risk-free proxy
# on or before an optional market-data date.
def _get_gbp_risk_free_rate(
    market_data_date=None,
):

    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id=IUDSOIA"
    )

    rate_data = pd.read_csv(
        url
    )

    selected_rate = _select_fred_observation(
        rate_data=rate_data,
        value_column="IUDSOIA",
        market_data_date=market_data_date,
        empty_error_message=(
            "No SONIA data available on or before "
            "the requested date."
        ),
    )

    return float(
        selected_rate
        / 100
    )


# Normalize Excel labels for robust column identification.
def _normalize_label(
    value,
):

    value = (
        str(value)
        .strip()
        .lower()
    )

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
        )
    )

    return value


# Find a column containing one of the required keywords.
def _find_column(
    columns,
    keywords,
):

    for column in columns:

        normalized_column = _normalize_label(
            column
        )

        if any(
            keyword in normalized_column
            for keyword in keywords
        ):

            return column

    return None


# Retrieve approximately one-year CHF
# Geldmarktbuchforderung yield.
#
# If a market-data date is supplied, only auctions on or before
# that date are considered.
def _get_chf_risk_free_rate(
    market_data_date=None,
):

    url = (
        "https://www.efv.admin.ch/dam/en/sd-web/"
        "d9FElh8dV0db/resultate-gmbf.xlsx"
    )

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=20,
    )

    response.raise_for_status()

    excel_content = response.content

    excel_file = pd.ExcelFile(
        BytesIO(
            excel_content
        )
    )

    suitable_data = []

    for sheet_name in excel_file.sheet_names:

        preview = pd.read_excel(
            BytesIO(
                excel_content
            ),
            sheet_name=sheet_name,
            header=None,
        )

        header_row = None

        for row_index in range(
            min(
                25,
                len(preview),
            )
        ):

            row_text = " ".join(
                _normalize_label(
                    value
                )
                for value in preview.iloc[
                    row_index
                ].dropna()
            )

            has_yield = (
                "yield" in row_text
                or "rendite" in row_text
            )

            has_maturity = (
                "maturity" in row_text
                or "verfall" in row_text
                or "falligkeit" in row_text
            )

            if (
                has_yield
                and has_maturity
            ):

                header_row = row_index
                break

        if header_row is None:
            continue

        data = pd.read_excel(
            BytesIO(
                excel_content
            ),
            sheet_name=sheet_name,
            header=header_row,
        )

        yield_column = _find_column(
            data.columns,
            [
                "yield",
                "rendite",
            ],
        )

        maturity_column = _find_column(
            data.columns,
            [
                "maturity",
                "verfall",
                "falligkeit",
            ],
        )

        auction_column = _find_column(
            data.columns,
            [
                "auction",
                "auktion",
            ],
        )

        # If auction date is unavailable,
        # use settlement/value date.
        if auction_column is None:

            auction_column = _find_column(
                data.columns,
                [
                    "settlement",
                    "valuta",
                    "value date",
                ],
            )

        if (
            yield_column is None
            or maturity_column is None
            or auction_column is None
        ):
            continue

        data = data.copy()

        data[
            "auction_date"
        ] = pd.to_datetime(
            data[
                auction_column
            ],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        data[
            "maturity_date"
        ] = pd.to_datetime(
            data[
                maturity_column
            ],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        yield_values = (
            data[
                yield_column
            ]
            .astype(str)
            .str.replace(
                ",",
                ".",
                regex=False,
            )
            .str.extract(
                r"(-?\d+(?:\.\d+)?)"
            )[0]
        )

        data[
            "yield"
        ] = pd.to_numeric(
            yield_values,
            errors="coerce",
        )

        data = data.dropna(
            subset=[
                "auction_date",
                "maturity_date",
                "yield",
            ]
        )

        if data.empty:
            continue

        data[
            "tenor_days"
        ] = (
            data[
                "maturity_date"
            ]
            - data[
                "auction_date"
            ]
        ).dt.days

        one_year_data = data[
            data[
                "tenor_days"
            ].between(
                300,
                400,
            )
        ].copy()

        if one_year_data.empty:
            continue

        suitable_data.append(
            one_year_data
        )

    if not suitable_data:

        raise ValueError(
            "No approximately 12-month CHF GMBF "
            "auction yield found."
        )

    gmbf_data = pd.concat(
        suitable_data,
        ignore_index=True,
    )

    requested_date = _normalize_market_data_date(
        market_data_date
    )

    if requested_date is not None:

        gmbf_data = gmbf_data[
            gmbf_data[
                "auction_date"
            ]
            <= requested_date
        ]

        if gmbf_data.empty:

            raise ValueError(
                "No approximately 12-month CHF GMBF auction "
                "yield is available on or before the requested date."
            )

    gmbf_data = gmbf_data.sort_values(
        "auction_date",
        ascending=False,
    )

    latest_date = gmbf_data[
        "auction_date"
    ].iloc[0]

    latest_auctions = gmbf_data[
        gmbf_data[
            "auction_date"
        ]
        == latest_date
    ].copy()

    latest_auctions[
        "distance_to_one_year"
    ] = (
        latest_auctions[
            "tenor_days"
        ]
        - 365
    ).abs()

    latest_row = (
        latest_auctions
        .sort_values(
            "distance_to_one_year"
        )
        .iloc[0]
    )

    latest_yield = latest_row[
        "yield"
    ]

    return float(
        latest_yield
        / 100
    )


# Retrieve the automatic rate for a supported currency.
#
# When no historical market-data date is supplied, preserve
# the original no-argument calls to the currency-specific
# functions. This keeps existing code and monkeypatched tests
# backward compatible.
def _get_live_risk_free_rate(
    currency,
    market_data_date=None,
):

    if currency == "USD":

        if market_data_date is None:

            return _get_usd_risk_free_rate()

        return _get_usd_risk_free_rate(
            market_data_date=market_data_date,
        )

    if currency == "EUR":

        if market_data_date is None:

            return _get_eur_risk_free_rate()

        return _get_eur_risk_free_rate(
            market_data_date=market_data_date,
        )

    if currency == "GBP":

        if market_data_date is None:

            return _get_gbp_risk_free_rate()

        return _get_gbp_risk_free_rate(
            market_data_date=market_data_date,
        )

    if currency == "CHF":

        if market_data_date is None:

            return _get_chf_risk_free_rate()

        return _get_chf_risk_free_rate(
            market_data_date=market_data_date,
        )

    raise ValueError(
        f"No automatic risk-free-rate source is configured "
        f"for currency '{currency}'."
    )


# Retrieve a risk-free fallback rate from the frozen Excel snapshot.
def _get_fallback_risk_free_rate(
    currency,
    market_data,
    tenor_years=1.0,
    market_data_date=None,
):

    if market_data is None:

        raise ValueError(
            "No fallback market data provided."
        )

    required_columns = [
        "data_type",
        "currency",
        "tenor_years",
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

    data[
        "data_type_normalized"
    ] = (
        data[
            "data_type"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data[
        "currency_normalized"
    ] = (
        data[
            "currency"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data[
        "tenor_numeric"
    ] = pd.to_numeric(
        data[
            "tenor_years"
        ],
        errors="coerce",
    )

    data[
        "value_numeric"
    ] = pd.to_numeric(
        data[
            "value"
        ],
        errors="coerce",
    )

    risk_free_data = data[
        data[
            "data_type_normalized"
        ]
        == "risk_free"
    ].copy()

    if risk_free_data.empty:

        raise ValueError(
            "No risk-free fallback data available."
        )

    # If a historical date was requested, do not allow
    # a fallback snapshot from a later date.
    _validate_fallback_snapshot_date(
        data=risk_free_data,
        market_data_date=market_data_date,
    )

    fallback_rows = risk_free_data[
        (
            risk_free_data[
                "currency_normalized"
            ]
            == currency
        )
        & (
            (
                risk_free_data[
                    "tenor_numeric"
                ]
                - float(
                    tenor_years
                )
            ).abs()
            <= 1e-10
        )
    ]

    if fallback_rows.empty:

        raise ValueError(
            f"No {tenor_years:g}-year risk-free fallback rate "
            f"found for currency '{currency}'."
        )

    if len(fallback_rows) != 1:

        raise ValueError(
            f"Multiple {tenor_years:g}-year risk-free fallback "
            f"rates found for currency '{currency}'."
        )

    fallback_rate = fallback_rows[
        "value_numeric"
    ].iloc[0]

    if pd.isna(
        fallback_rate
    ):

        raise ValueError(
            f"Invalid risk-free fallback rate "
            f"for currency '{currency}'."
        )

    return float(
        fallback_rate
    )


# Retrieve the risk-free rate for a given currency.
#
# Automatic market data is always attempted first.
# The frozen Excel snapshot is used only if automatic retrieval fails.
#
# If market_data_date is specified, the latest available automatic
# observation on or before that date is used.
def get_risk_free_rate(
    currency,
    market_data=None,
    tenor_years=1.0,
    return_source=False,
    market_data_date=None,
):

    currency = (
        str(currency)
        .upper()
        .strip()
    )

    if not currency:

        raise ValueError(
            "Currency must not be empty."
        )

    requested_date = _normalize_market_data_date(
        market_data_date
    )

    try:

        # Preserve the previous call signature when the
        # latest observation is requested.
        if requested_date is None:

            rate = _get_live_risk_free_rate(
                currency
            )

        else:

            rate = _get_live_risk_free_rate(
                currency=currency,
                market_data_date=requested_date,
            )

        source = "automatic_market_data"

    except Exception as live_error:

        # Without a supplied fallback snapshot, preserve the
        # previous behavior and propagate the live-data error.
        if market_data is None:

            raise live_error

        try:

            rate = _get_fallback_risk_free_rate(
                currency=currency,
                market_data=market_data,
                tenor_years=tenor_years,
                market_data_date=requested_date,
            )

            source = "fallback_snapshot"

        except Exception as fallback_error:

            raise ValueError(
                f"Could not obtain a risk-free rate for "
                f"currency '{currency}'. "
                f"Automatic retrieval failed with: "
                f"{live_error}. "
                f"Fallback retrieval failed with: "
                f"{fallback_error}."
            ) from fallback_error

    if return_source:

        return rate, source

    return rate