# Volatility Model Calibration Pipeline

A Python-based quantitative finance project for option pricing, implied-volatility inversion, synthetic volatility-surface generation, and Heston model calibration experiments.

This project is built as a modular calibration pipeline rather than a single notebook. It includes tested pricing routines, implied-volatility inversion, calibration metrics, bounded least-squares optimisation, a Heston characteristic-function pricer, synthetic volatility-surface visualisation, and a synthetic Heston recovery experiment.

---

## Project Motivation

Volatility models are used in quantitative finance to explain and reproduce the structure of option prices across strikes and maturities. In practice, traders and quants often work with implied-volatility surfaces rather than raw option prices.

This project explores the workflow:

```text
model parameters
-> option prices
-> implied volatilities
-> residuals against a target surface
-> calibration error
-> optimiser update
## Heston Calibration Robustness Diagnostics

To test whether the Heston calibration routine depends on a single favourable initial guess, the project runs a robustness experiment across multiple starting parameter vectors.

Each run records the initial calibration error and final calibration error after bounded least-squares optimisation. The diagnostic plots below compare the initial and final RMSE/SSE values across robustness runs.

![Heston robustness RMSE](results/figures/heston_robustness_rmse.png)

![Heston robustness SSE](results/figures/heston_robustness_sse.png)

The sharp reduction in error demonstrates that the optimiser can substantially improve the calibration fit from different initial guesses. The experiment also highlights an important practical issue in stochastic-volatility calibration: different parameter vectors may produce similar implied-volatility surfaces, so calibration should be assessed using both error reduction and parameter sensitivity.

