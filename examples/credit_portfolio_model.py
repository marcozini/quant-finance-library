#### Credit Portfolio Model Example ####

from quant_finance.credit.portfolio_model import run_credit_portfolio_model


FILE_PATH = "data/credit_risk_input.xlsx"

results = run_credit_portfolio_model(FILE_PATH)

portfolio = results["portfolio"]
summary = results["summary"]


print("\nCredit portfolio results:")
print(
    portfolio[
        [
            "counterparty",
            "ticker",
            "pd_method",
            "exposure",
            "lgd",
            "pd",
            "pd_source",
            "expected_loss",
        ]
    ].to_string(index=False)
)


print("\nFactor loadings:")
print(
    portfolio[
        [
            "counterparty",
            "factor_calibration_ticker",
            "factor_source",
            "global_loading",
            "region_loading",
            "sector_loading",
            "idiosyncratic_loading",
        ]
    ].to_string(index=False)
)


print(f"\nDefault expected loss: {summary['default_expected_loss']:,.4f}")
print(f"Migration mean loss: {summary['migration_mean_loss']:,.4f}")
print(f"{summary['confidence_level']:.1%} VaR: {summary['value_at_risk']:,.4f}")
print(
    f"{summary['confidence_level']:.1%} Expected Shortfall: "
    f"{summary['expected_shortfall']:,.4f}"
)
print(f"Unexpected loss: {summary['unexpected_loss']:,.4f}")


print("\nCounterparty risk contributions:")
print(
    portfolio[
        [
            "counterparty",
            "exposure",
            "incremental_var",
            "incremental_es",
            "marginal_var",
            "marginal_es",
            "standalone_var",
            "standalone_es",
            "var_diversification_benefit",
            "es_diversification_benefit",
        ]
    ].to_string(index=False)
)


print(
    f"\nMaximum simulated portfolio loss: "
    f"{summary['maximum_simulated_loss']:,.4f}"
)

print(
    f"Minimum simulated portfolio loss: "
    f"{summary['minimum_simulated_loss']:,.4f}"
)