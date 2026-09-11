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