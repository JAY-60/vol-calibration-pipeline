# Heston Model Calibration Notes

This note explains the mathematical and numerical ideas behind the volatility calibration pipeline.

The project implements a Python workflow for option pricing, implied-volatility inversion, Heston model calibration, and robustness diagnostics across multiple starting guesses.

The main purpose is to test whether the calibration routine is accurate, reproducible, and robust enough to be inspected.

---

## 1. Model Overview

The Heston model is a stochastic volatility model. Unlike Black-Scholes, where volatility is constant, Heston allows the variance of the asset to evolve randomly over time.

Under the risk-neutral measure, the model is commonly written as:

```text
dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S

dv_t = kappa(theta - v_t)dt + sigma sqrt(v_t)dW_t^v

dW_t^S dW_t^v = rho dt
```

where:

```text
S_t     = asset price
v_t     = instantaneous variance
r       = risk-free rate
kappa   = speed of variance mean reversion
theta   = long-run variance
sigma   = volatility of volatility
rho     = correlation between asset and variance shocks
v0      = initial variance
```

The calibrated parameter vector is:

```text
[kappa, theta, sigma, rho, v0]
```

The reason Heston is useful is that it can produce implied-volatility smiles and skews, which constant-volatility Black-Scholes cannot naturally reproduce.

---

## 2. Parameter Interpretation

`kappa` controls how quickly variance returns to its long-run level. A higher value means volatility shocks decay more quickly.

`theta` is the long-run variance level. Since volatility is the square root of variance, a higher `theta` usually means a higher long-run volatility level.

`sigma` is the volatility of volatility. It controls how strongly the variance process fluctuates.

`rho` is the correlation between asset-price shocks and variance shocks. In equity markets, this is often negative because falling prices are commonly associated with rising volatility.

`v0` is the initial variance. It is especially important for short-maturity options.

These parameters interact with each other, so calibration is not always unique. Different parameter combinations can sometimes produce similar implied-volatility surfaces.

---

## 3. Implied Volatility and Calibration

Options are often compared using implied volatility rather than raw price.

Implied volatility is the Black-Scholes volatility input that reproduces a given option price.

The project uses the following workflow:

```text
Heston parameters
-> Heston option prices
-> Black-Scholes implied volatilities
-> comparison with target implied volatilities
-> calibration error
```

This means calibration is treated as an inverse problem.

Pricing is the forward problem:

```text
given model parameters -> compute option prices
```

Calibration is the inverse problem:

```text
given target implied volatilities -> infer model parameters
```

For each option quote, the residual is:

```text
residual = model implied volatility - target implied volatility
```

The optimiser tries to make these residuals as small as possible.

---

## 4. Optimisation Objective

The project uses bounded nonlinear least-squares optimisation.

The objective is:

```text
minimise sum((model implied volatility - target implied volatility)^2)
```

Bounds are used because the optimiser should not search over invalid or unstable parameter regions.

For example:

```text
variance parameters should be positive
rho must lie between -1 and 1
extreme parameters can make the Heston pricer unstable
```

The bounds help keep the calibration numerically and financially reasonable.

---

## 5. Error Metrics

The project records several calibration error metrics.

### SSE

SSE means sum of squared errors:

```text
SSE = sum(residual^2)
```

This is the total squared calibration error.

### RMSE

RMSE means root mean squared error:

```text
RMSE = sqrt(mean(residual^2))
```

RMSE is useful because it is measured on the same scale as implied volatility.

As a rough guide:

```text
RMSE = 0.01    means about 1 volatility point
RMSE = 0.001   means about 0.1 volatility points
RMSE = 0.0001  means about 0.01 volatility points
```

### MAE

MAE means mean absolute error:

```text
MAE = mean(abs(residual))
```

MAE measures the average absolute size of the calibration error.

---

## 6. Function Evaluations

`nfev` means number of function evaluations.

In this project, one function evaluation means:

```text
try one candidate Heston parameter vector
-> price options under Heston
-> convert prices to implied volatilities
-> compute residuals
-> return the error to the optimiser
```

`max_nfev` is the maximum number of function evaluations allowed before the optimiser stops.

This is the optimiser's computational budget.

The robustness experiment originally used a budget of 100 evaluations. Increasing this to 300 improved convergence from:

```text
18 successful calibrations out of 25
```

to:

```text
23 successful calibrations out of 25
```

This showed that some earlier failures were caused by the optimiser stopping too early.

---

## 7. Robustness Experiment

A single successful calibration does not prove that the calibration routine is robust.

The project therefore runs a seeded multi-start robustness experiment.

“Seeded” means that the starting guesses vary, but the randomness is controlled so the experiment can be reproduced.

The purpose is to test:

```text
Does calibration still work when started from different initial guesses?
```

The current result is:

```text
23 successful calibrations
2 unsuccessful calibrations
```

This suggests that the calibration pipeline is robust for most tested starting points.

The remaining failures are not hidden. They are useful because they identify difficult regions of the Heston calibration landscape.

---

## 8. Numerical Warnings

Heston pricing through characteristic functions involves complex-valued expressions, logarithms, divisions, and numerical integration.

For difficult parameter combinations, the pricer may produce warnings such as:

```text
divide by zero encountered
invalid value encountered
invalid value encountered in log
roundoff error in numerical integration
```

These warnings usually indicate numerical stress.

They do not automatically mean the project is broken. They show that some candidate parameter regions are fragile and need to be monitored.

This is why the project uses diagnostics instead of relying on one successful calibration.

---

## 9. Current Limitations

The current target volatility surface is synthetic rather than based on live market data.

The robustness experiment uses 25 starting guesses, not an exhaustive search.

The Heston pricer can still produce numerical warnings for difficult parameter regions.

The project does not yet include real option-chain ingestion.

The project does not yet compare characteristic-function prices against Monte Carlo prices.

The implementation is Python-based and is not intended as a low-latency production pricing system.

---

## 10. Possible Extensions

Natural next steps include:

```text
vectorised Heston Monte Carlo simulation
Euler-Maruyama simulation of Heston paths
comparison of Monte Carlo prices with characteristic-function prices
real option-chain ingestion
parameter-stability diagnostics
GitHub Actions for automated testing
performance benchmarking
selected C++ pricing routines
```

The most natural next extension is vectorised Heston Monte Carlo simulation, because it would allow the project to compare simulation-based pricing with characteristic-function pricing.

---

## Summary

This project treats Heston calibration as a numerical inverse problem.

The pipeline starts with a target implied-volatility surface and searches for Heston parameters that reproduce it.

The calibration objective minimises implied-volatility residuals.

The robustness experiment shows that increasing the optimiser budget from 100 to 300 function evaluations improved convergence from 18/25 to 23/25 successful runs.

The key lesson is:

```text
A calibration result is only useful if it is accurate, reproducible, and understood.
```

The project is therefore not only about implementing the Heston model. It is about building a tested, explainable calibration workflow with clear diagnostics and honest limitations.