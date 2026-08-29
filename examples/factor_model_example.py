#### Credit Factor Model Example ####

from datetime import date

from quant_finance.credit.factor_market_data import (
    estimate_market_factor_loadings,
)


# Estimate Siemens factor loadings using three years of market data.

factor_loadings = estimate_market_factor_loadings(
    company_ticker="SIE.DE",
    global_ticker="ACWI",
    region_ticker="VGK",
    sector_ticker="EXI",
    start_date="2023-08-29",
    end_date=date.today(),
)


# Display estimated factor loadings.

print("\nSiemens factor loadings:")

print(
    f"Global loading: "
    f"{factor_loadings['global_loading']:.4f}"
)

print(
    f"Region loading: "
    f"{factor_loadings['region_loading']:.4f}"
)

print(
    f"Sector loading: "
    f"{factor_loadings['sector_loading']:.4f}"
)

print(
    f"Idiosyncratic loading: "
    f"{factor_loadings['idiosyncratic_loading']:.4f}"
)