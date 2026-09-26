# Quant Finance Python Library

A modular Python project for quantitative finance, covering option pricing, volatility, numerical methods, hedging, structural credit risk, credit migration, dependence modeling, and portfolio tail risk.

The project emphasizes transparent quantitative logic, reusable code, testing, validation, and a clear separation between core models and analysis workflows.

---

## Project Overview

The repository covers four main areas of quantitative finance.

### 1. Option Pricing and Implied Volatility

- Black–Scholes pricing for European calls and puts
- Analytical Greeks
- Implied volatility using bisection and Newton–Raphson
- Put-call parity
- Volatility smile and surface analysis using market option data

### 2. Numerical Pricing and Hedging

- Monte Carlo pricing under risk-neutral geometric Brownian motion
- Simulation error and convergence analysis
- Binomial tree pricing
- European, American, and Bermudan exercise styles
- Delta hedging simulation and hedging P&L
- Comparison of Black–Scholes, Monte Carlo, and binomial pricing methods

### 3. Structural Credit Risk

- Merton structural credit model
- Numerical calibration of firm asset value and asset volatility
- Structural probability of default
- Rating-based and Merton-based credit inputs
- Credit rating migration and spread-based bond revaluation

### 4. Credit Portfolio Risk

The credit portfolio component combines the earlier modeling elements into a configurable portfolio risk framework.

It includes:

- multi-currency portfolio exposures
- rating and structural probabilities of default
- one-year rating migration
- global, regional, and sector risk factors
- Gaussian and Student t-copula dependence
- Monte Carlo portfolio loss simulation
- Value at Risk and Expected Shortfall
- standalone, incremental, and exposure-normalized counterparty risk measures
- Excel-based portfolio and model configuration
- market data retrieval and fallback handling
- automated output generation
- model validation and sensitivity analysis

---

## Credit Portfolio Validation

The credit portfolio model includes a separate validation framework rather than relying only on a single baseline run.

The analysis challenges:

- independent, Gaussian, and Student t-copula dependence
- Student t-copula degrees of freedom
- LGD assumptions
- single-counterparty concentration
- systematic factor structure
- Monte Carlo sample size and random seed
- VaR and Expected Shortfall confidence levels

For the demonstration portfolio, dependence assumptions are among the strongest drivers of tail risk. The concentration analysis also highlights an important distinction between VaR and Expected Shortfall: a portfolio can exhibit a lower VaR while still generating substantially more severe losses further into the tail.

All portfolio positions, ratings, LGDs, bond characteristics, and selected credit inputs are demonstration data and should not be interpreted as actual company exposures or external credit assessments.

---

## Example Baseline Results

The baseline credit portfolio configuration uses a 99.5% confidence level, 100,000 Monte Carlo scenarios, and a Student t-copula with five degrees of freedom.

| Metric | Result |
|---|---:|
| Portfolio exposure | USD 899.59m |
| Default expected loss | USD 1.17m |
| Mean simulated loss | USD 5.14m |
| 99.5% VaR | USD 78.31m |
| 99.5% Expected Shortfall | USD 141.13m |

The difference between default expected loss and the complete simulated loss distribution reflects the additional effects of rating migration, spread revaluation, concentration, and systematic dependence.

---

## Technical Report

A detailed technical report accompanies the credit portfolio model:

**[Credit Portfolio Risk Analytics](docs/Credit_Portfolio_Risk_Analytics.pdf)**

The report documents the methodology, implementation architecture, market data treatment, structural Merton calibration, rating migration, systematic factor modeling, copula dependence, portfolio risk measures, model validation, sensitivity analysis, and model limitations.

---

## Repository Structure

```text
quant-finance-library/
├── quant_finance/
│   ├── options/
│   ├── credit/
│   └── market_data/
├── examples/
├── notebooks/
├── tests/
├── data/
├── docs/
│   └── Credit_Portfolio_Risk_Analytics.pdf
├── outputs/
├── pyproject.toml
└── README.md
```

- `quant_finance/` contains reusable quantitative model and market-data code.
- `examples/` contains executable demonstrations and portfolio workflows.
- `notebooks/` contains exploratory and explanatory analyses.
- `tests/` contains the automated `pytest` test suite.
- `data/` contains local steering and model input files.
- `docs/` contains the technical project documentation.
- `outputs/` contains generated model results, reports, plots, and validation outputs.

---

## Installation

Python 3.10 or later is required.

Clone the repository and move into the project directory:

```bash
git clone https://github.com/marcozini/quant-finance-library.git
cd quant-finance-library
```

Install the project together with the development, testing, and notebook dependencies:

```bash
pip install -e ".[dev,notebooks]"
```

The required dependencies and optional development and notebook dependencies are defined in `pyproject.toml`.

The editable installation (`-e`) allows changes to the source code to be used immediately without reinstalling the package.

---

## Running the Project

Run the automated test suite:

```bash
pytest -q
```

Run the credit portfolio model:

```bash
python examples/credit_portfolio_output.py
```

Run the credit portfolio validation framework:

```bash
python examples/credit_portfolio_validation.py
```

Additional option-pricing, volatility, Monte Carlo, binomial-tree, Merton-model, market-data, and hedging examples are available in the `examples/` directory.

Exploratory analyses, including volatility smile and surface analysis and the credit portfolio model notebook, are available in the `notebooks/` directory.

---

## Testing

The project contains an automated `pytest` test suite covering:

- Black–Scholes pricing
- Greeks
- implied volatility
- Monte Carlo pricing
- binomial-tree pricing
- delta hedging
- bond valuation
- Merton structural credit risk
- Merton calibration
- rating and migration logic
- systematic factor modeling
- market data handling
- dependence modeling
- portfolio simulation
- portfolio risk analytics
- model inputs and validation logic

Current project status:

**213 tests passed.**

---

## Model Scope and Limitations

The project is a quantitative modeling framework rather than a production credit risk system.

Important simplifications include synthetic rating transition and spread assumptions, deterministic LGD, simplified structural default modeling, fixed exposures, proxy-based factor calibration, simplified interest-rate treatment, and limited historical backtesting.

These assumptions are made explicit in the technical report and are deliberately included in the model-risk and validation discussion.

---

## Motivation and Further Development

This project was developed as a hands-on quantitative finance learning project, with the aim of connecting financial theory, numerical methods, software implementation, testing, and model validation within a single codebase.

A particular area of interest is dependence modeling, especially correlations, copulas, and the behavior of portfolio risk under common systematic shocks. This motivated the extension from individual structural credit models toward a portfolio framework combining rating migration, systematic factors, and Student t-copula tail dependence.

Potential future extensions include empirical rating transition and credit spread data, stochastic LGD, richer structural credit models, improved term-structure modeling, empirical dependence calibration, alternative copula structures, stress testing, and historical backtesting.

The aim is to keep the models understandable, testable, and quantitatively transparent rather than adding complexity for its own sake.
