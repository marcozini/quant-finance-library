#### Risk-Free Rates ####

import unicodedata
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


# Retrieve the latest available one-year US Treasury yield.

def _get_usd_risk_free_rate():

    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id=DGS1"
    )

    rate_data = pd.read_csv(url)
    rate_data["DGS1"] = pd.to_numeric(rate_data["DGS1"], errors="coerce")
    rate_data = rate_data.dropna(subset=["DGS1"])

    if rate_data.empty:
        raise ValueError(
            "No one-year US Treasury yield data available."
        )

    latest_yield = rate_data["DGS1"].iloc[-1]

    return float(latest_yield / 100)


# Retrieve the latest available one-year euro area AAA spot rate.

def _get_eur_risk_free_rate():

    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y"
    )

    response = requests.get(
        url,
        params={
            "format": "csvdata",
            "lastNObservations": 1,
        },
        timeout=15,
    )

    response.raise_for_status()

    rate_data = pd.read_csv(
        StringIO(response.text)
    )

    if rate_data.empty:
        raise ValueError(
            "No one-year EUR spot-rate data available."
        )

    latest_yield = pd.to_numeric(
        rate_data["OBS_VALUE"],
        errors="coerce",
    ).dropna()

    if latest_yield.empty:
        raise ValueError(
            "No valid one-year EUR spot rate available."
        )

    return float(latest_yield.iloc[-1] / 100)


# Retrieve GBP short-term risk-free proxy.

def _get_gbp_risk_free_rate():

    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id=IUDSOIA"
    )

    rate_data = pd.read_csv(url)

    rate_data["IUDSOIA"] = pd.to_numeric(
        rate_data["IUDSOIA"],
        errors="coerce",
    )

    rate_data = rate_data.dropna(
        subset=["IUDSOIA"]
    )

    if rate_data.empty:
        raise ValueError(
            "No SONIA data available."
        )

    latest_rate = rate_data["IUDSOIA"].iloc[-1]

    return float(latest_rate / 100)


# Normalize Excel labels for robust column identification.

def _normalize_label(value):

    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    return value


# Find a column containing one of the required keywords.

def _find_column(columns, keywords):

    for column in columns:

        normalized_column = _normalize_label(column)

        if any(
            keyword in normalized_column
            for keyword in keywords
        ):
            return column

    return None


# Retrieve approximately one-year CHF Geldmarktbuchforderung yield.

def _get_chf_risk_free_rate():

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
        BytesIO(excel_content)
    )

    suitable_data = []

    for sheet_name in excel_file.sheet_names:

        preview = pd.read_excel(
            BytesIO(excel_content),
            sheet_name=sheet_name,
            header=None,
        )

        header_row = None

        for row_index in range(
            min(25, len(preview))
        ):

            row_text = " ".join(
                _normalize_label(value)
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

            if has_yield and has_maturity:
                header_row = row_index
                break

        if header_row is None:
            continue

        data = pd.read_excel(
            BytesIO(excel_content),
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

        # If auction date is unavailable, use settlement/value date.
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

        data["auction_date"] = pd.to_datetime(
            data[auction_column],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        data["maturity_date"] = pd.to_datetime(
            data[maturity_column],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        yield_values = (
            data[yield_column]
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

        data["yield"] = pd.to_numeric(
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

        data["tenor_days"] = (
            data["maturity_date"]
            - data["auction_date"]
        ).dt.days

        one_year_data = data[
            data["tenor_days"].between(
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

    gmbf_data = gmbf_data.sort_values(
        "auction_date",
        ascending=False,
    )

    latest_date = gmbf_data[
        "auction_date"
    ].iloc[0]

    latest_auctions = gmbf_data[
        gmbf_data["auction_date"]
        == latest_date
    ].copy()

    latest_auctions["distance_to_one_year"] = (
        latest_auctions["tenor_days"]
        - 365
    ).abs()

    latest_row = latest_auctions.sort_values(
        "distance_to_one_year"
    ).iloc[0]

    latest_yield = latest_row["yield"]

    return float(latest_yield / 100)


# Retrieve the live/automatic rate for a supported currency.

def _get_live_risk_free_rate(currency):

    if currency == "USD":
        return _get_usd_risk_free_rate()

    if currency == "EUR":
        return _get_eur_risk_free_rate()

    if currency == "GBP":
        return _get_gbp_risk_free_rate()

    if currency == "CHF":
        return _get_chf_risk_free_rate()

    raise ValueError(
        f"No automatic risk-free-rate source is configured "
        f"for currency '{currency}'."
    )


# Retrieve a risk-free fallback rate from the frozen Excel snapshot.

def _get_fallback_risk_free_rate(
    currency,
    market_data,
    tenor_years=1.0,
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

    data["tenor_numeric"] = pd.to_numeric(
        data["tenor_years"],
        errors="coerce",
    )

    data["value_numeric"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    fallback_rows = data[
        (data["data_type_normalized"] == "risk_free")
        & (data["currency_normalized"] == currency)
        & (
            (
                data["tenor_numeric"]
                - float(tenor_years)
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

    if pd.isna(fallback_rate):
        raise ValueError(
            f"Invalid risk-free fallback rate "
            f"for currency '{currency}'."
        )

    return float(fallback_rate)


# Retrieve the risk-free rate for a given currency.
#
# Automatic market data is always attempted first.
# The frozen Excel snapshot is used only if automatic retrieval fails.

def get_risk_free_rate(
    currency,
    market_data=None,
    tenor_years=1.0,
    return_source=False,
):

    currency = str(
        currency
    ).upper().strip()

    if not currency:
        raise ValueError(
            "Currency must not be empty."
        )

    try:

        rate = _get_live_risk_free_rate(
            currency
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