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