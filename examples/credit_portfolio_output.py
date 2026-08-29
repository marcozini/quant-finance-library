#### Credit Portfolio Model Output ####

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Font

from quant_finance.credit.portfolio_model import run_credit_portfolio_model


FILE_PATH = "data/credit_risk_input.xlsx"
OUTPUT_DIR = Path("outputs")
OUTPUT_EXCEL = OUTPUT_DIR / "credit_portfolio_results.xlsx"
OUTPUT_PLOT = OUTPUT_DIR / "credit_loss_distribution.png"


def create_loss_distribution_plot(portfolio_losses, summary):

    mean_loss = summary["migration_mean_loss"]
    var = summary["value_at_risk"]
    es = summary["expected_shortfall"]
    confidence = summary["confidence_level"]

    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_losses, bins=100, alpha=0.8)
    plt.axvline(mean_loss, linestyle="--", label=f"Mean loss: {mean_loss:.2f}")
    plt.axvline(var, linestyle="--", label=f"{confidence:.1%} VaR: {var:.2f}")
    plt.axvline(es, linestyle=":", label=f"{confidence:.1%} ES: {es:.2f}")

    plt.title("Simulated Credit Portfolio Loss Distribution")
    plt.xlabel("Portfolio loss")
    plt.ylabel("Number of simulations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()


def format_excel_output():

    workbook = load_workbook(OUTPUT_EXCEL)

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        sheet.freeze_panes = "A2"

        for cell in sheet[1]:
            cell.font = Font(bold=True)

    summary_sheet = workbook["portfolio_summary"]
    summary_sheet.column_dimensions["A"].width = 32
    summary_sheet.column_dimensions["B"].width = 18

    counterparty_sheet = workbook["counterparty_results"]

    for column in counterparty_sheet.columns:
        letter = column[0].column_letter
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column[:100])
        counterparty_sheet.column_dimensions[letter].width = min(max_length + 2, 30)

    settings_sheet = workbook["model_settings"]
    settings_sheet.column_dimensions["A"].width = 30
    settings_sheet.column_dimensions["B"].width = 25

    scenario_sheet = workbook["scenario_losses"]
    scenario_sheet.column_dimensions["A"].width = 12
    scenario_sheet.column_dimensions["B"].width = 18

    image = Image(OUTPUT_PLOT)
    image.width = 750
    image.height = 450
    summary_sheet.add_image(image, "D2")

    workbook.save(OUTPUT_EXCEL)


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    results = run_credit_portfolio_model(FILE_PATH)

    portfolio = results["portfolio"]
    settings = results["settings"]
    factor_proxies = results["factor_proxies"]
    rating_migration_matrix = results["rating_migration_matrix"]
    portfolio_losses = results["portfolio_losses"]
    summary = results["summary"]

    confidence = summary["confidence_level"]

    summary_df = pd.DataFrame({
        "metric": [
            "Default expected loss",
            "Migration mean loss",
            f"{confidence:.1%} VaR",
            f"{confidence:.1%} Expected Shortfall",
            "Unexpected loss",
            "Maximum simulated loss",
            "Minimum simulated loss",
        ],
        "value": [
            summary["default_expected_loss"],
            summary["migration_mean_loss"],
            summary["value_at_risk"],
            summary["expected_shortfall"],
            summary["unexpected_loss"],
            summary["maximum_simulated_loss"],
            summary["minimum_simulated_loss"],
        ],
    })

    counterparty_columns = [
        "counterparty",
        "ticker",
        "factor_loading_proxy_ticker",
        "pd_method",
        "rating",
        "exposure",
        "currency",
        "lgd",
        "pd",
        "pd_source",
        "expected_loss",
        "equity_value",
        "equity_volatility",
        "debt",
        "asset_value",
        "asset_volatility",
        "risk_free_rate",
        "maturity",
        "coupon_rate",
        "payment_frequency",
        "sector",
        "region",
        "factor_calibration_ticker",
        "factor_source",
        "global_loading",
        "region_loading",
        "sector_loading",
        "idiosyncratic_loading",
        "incremental_var",
        "incremental_es",
        "marginal_var",
        "marginal_es",
        "standalone_var",
        "standalone_es",
        "var_diversification_benefit",
        "es_diversification_benefit",
    ]

    counterparty_results = portfolio[counterparty_columns].copy()
    settings_df = pd.DataFrame(settings.items(), columns=["setting", "value"])

    scenario_losses = pd.DataFrame({
        "scenario": range(1, len(portfolio_losses) + 1),
        "portfolio_loss": portfolio_losses,
    })

    create_loss_distribution_plot(portfolio_losses, summary)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="portfolio_summary", index=False)
        counterparty_results.to_excel(writer, sheet_name="counterparty_results", index=False)
        settings_df.to_excel(writer, sheet_name="model_settings", index=False)
        scenario_losses.to_excel(writer, sheet_name="scenario_losses", index=False)
        factor_proxies.to_excel(writer, sheet_name="factor_proxies", index=False)
        rating_migration_matrix.to_excel(writer, sheet_name="rating_migration_matrix", index=False)

    format_excel_output()

    print(f"\nExcel output created: {OUTPUT_EXCEL}")
    print(f"Loss distribution created: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()