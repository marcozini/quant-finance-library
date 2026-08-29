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
# Note:
# GBP currently uses an overnight proxy rather than a true
# one-year zero-coupon rate.


# Retrieve the latest available one-year US Treasury yield.

def _get_usd_risk_free_rate():
    """
    Retrieve the latest available one-year US Treasury yield
    from Federal Reserve Economic Data (FRED).

    Returns
    -------
    float
        Risk-free rate expressed as a decimal.
    """

    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id=DGS1"
    )

    rate_data = pd.read_csv(url)

    rate_data["DGS1"] = pd.to_numeric(
        rate_data["DGS1"],
        errors="coerce",
    )

    rate_data = rate_data.dropna(
        subset=["DGS1"]
    )

    if rate_data.empty:
        raise ValueError(
            "No one-year US Treasury yield data available."
        )

    latest_yield = rate_data[
        "DGS1"
    ].iloc[-1]

    return float(
        latest_yield / 100
    )


# Retrieve the latest available one-year euro area AAA spot rate.

def _get_eur_risk_free_rate():
    """
    Retrieve the latest available one-year euro area AAA
    government bond spot rate from the ECB.

    Returns
    -------
    float
        Risk-free rate expressed as a decimal.
    """

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

    return float(
        latest_yield.iloc[-1] / 100
    )


# Retrieve GBP short-term risk-free proxy.

def _get_gbp_risk_free_rate():
    """
    Retrieve the latest available SONIA rate from FRED
    as a short-term GBP risk-free-rate proxy.

    Notes
    -----
    SONIA is an overnight rate and therefore not a true
    one-year zero-coupon rate.

    Returns
    -------
    float
        Risk-free-rate proxy expressed as a decimal.
    """

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

    latest_rate = rate_data[
        "IUDSOIA"
    ].iloc[-1]

    return float(
        latest_rate / 100
    )


# Normalize Excel labels for robust column identification.

def _normalize_label(value):

    value = str(value).strip().lower()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    return value


# Find a column containing one of the required keywords.

def _find_column(columns, keywords):

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


# Retrieve approximately one-year CHF Geldmarktbuchforderung yield.

def _get_chf_risk_free_rate():
    """
    Retrieve the latest approximately 12-month yield of a
    Geldmarktbuchforderung der Schweizerischen Eidgenossenschaft
    (GMBF).

    GMBF are short-term debt instruments issued by the Swiss
    Confederation, typically with maturities between three
    and twelve months.

    Returns
    -------
    float
        Risk-free rate expressed as a decimal.
    """

    url = (
        "https://www.efv.admin.ch/dam/en/sd-web/"
        "d9FElh8dV0db/resultate-gmbf.xlsx"
    )

    # Download official auction-results Excel file.
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=20,
    )

    response.raise_for_status()

    excel_content = response.content

    # Identify available Excel sheets.
    excel_file = pd.ExcelFile(
        BytesIO(excel_content)
    )

    suitable_data = []

    for sheet_name in excel_file.sheet_names:

        # First read without assuming where the header begins.
        preview = pd.read_excel(
            BytesIO(excel_content),
            sheet_name=sheet_name,
            header=None,
        )

        header_row = None

        # Search the first rows for the actual table header.
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

        # Read the sheet again using the detected header.
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

        # Parse mixed date formats from the official Excel file.
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

        # Convert yield values to numeric percentage values.
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

        # Calculate original maturity in days.
        data["tenor_days"] = (
            data["maturity_date"]
            - data["auction_date"]
        ).dt.days

        # Keep instruments close to a one-year maturity.
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

    # Combine suitable observations from all sheets.
    gmbf_data = pd.concat(
        suitable_data,
        ignore_index=True,
    )

    # Prefer the most recent auction.
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

    # If several instruments exist,
    # choose the one closest to 365 days.
    latest_auctions["distance_to_one_year"] = (
        latest_auctions["tenor_days"]
        - 365
    ).abs()

    latest_row = latest_auctions.sort_values(
        "distance_to_one_year"
    ).iloc[0]

    latest_yield = latest_row[
        "yield"
    ]

    # Auction yields in the source are expressed in percent.
    return float(
        latest_yield / 100
    )


# Retrieve the risk-free rate for a given currency.

def get_risk_free_rate(currency):
    """
    Retrieve the latest available risk-free rate
    for the specified currency.

    Parameters
    ----------
    currency : str
        Currency code: USD, EUR, GBP, or CHF.

    Returns
    -------
    float
        Risk-free rate expressed as a decimal.
    """

    currency = currency.upper().strip()

    if currency == "USD":
        return _get_usd_risk_free_rate()

    if currency == "EUR":
        return _get_eur_risk_free_rate()

    if currency == "GBP":
        return _get_gbp_risk_free_rate()

    if currency == "CHF":
        return _get_chf_risk_free_rate()

    raise ValueError(
        f"Unsupported currency '{currency}'. "
        f"Supported currencies are USD, EUR, GBP, and CHF."
    )