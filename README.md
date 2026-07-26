

[![PyPI](https://img.shields.io/pypi/v/nowcasting-dfm)](https://img.shields.io/pypi/v/nowcasting-dfm) 
![t](https://img.shields.io/badge/status-maintained-yellow.svg) [![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![CI / CD](https://github.com/SermetPekin/nowcasting-dfm/actions/workflows/ci.yml/badge.svg?1)](https://github.com/SermetPekin/nowcasting-dfm/actions/workflows/ci.yml?1)



# nowcasting-dfm

A Python implementation of the Dynamic Factor Model (DFM) for macroeconomic nowcasting, extending the FRBNY framework (Qian & Bok) with a modern API: real-time vintage management, Kalman-based news decomposition, optional Numba acceleration, caching, and interactive Plotly visualizations.

Based on "[Macroeconomic Nowcasting and Forecasting with Big Data](https://www.newyorkfed.org/research/staff_reports/sr830.html)" (Bok et al., *Staff Reports 830*, NY Fed).

## Installation

```bash
pip install nowcasting-dfm
```

For Jupyter notebook support:

```bash
pip install "nowcasting-dfm[all]"
```

## Quick Start

```python
from dfm_sp import Options, run_with_options, run, download_sample_data

# Download sample US vintage data (only needed once)
download_sample_data()

# Configure and run the model
options = Options(
    vintage="2016-12-16",
    country="US",
    spec_file_name="Spec_US_example.xls",
    max_iter=5000,
    use_cache=True,
)

Spec, X, Time, Z = run_with_options(options)
result = run(X, Spec, options)
```

### Nowcast update — decompose the impact of new data releases

```python
from dfm_sp import Options, sp_update_nowcast

options = Options(vintage="2016-12-16", country="US", spec_file_name="Spec_US_example.xls")

result = sp_update_nowcast(
    options,
    new_date="2016-12-23",
    series="GDPC1",
    period="2016q4",
)
result["fig"].show()  # interactive Plotly waterfall chart
```

## Architecture & Optimizations

*   **Modular Architecture (`sp_*` modules):** Transitioned the procedural scripts into a structured object-oriented library (`dfm_sp`). Introduced formal configuration dataclasses (`sp_classes.py`) and modular plotting logic (`sp_plots.py`) to streamline experimental workflows.
*   **Centralized Transformations (`sp_transformations.py`):** Abstracted legacy nested-lambda blocks into a dedicated `MacroTransformations` registry. Implements 4 new stationary bounds (`dln`, `dl1`, `d2l`, `zsc`) allowing complex structural vector geometries without inline array indexing side-effects.
*   **Execution Caching:** Integrated a `use_cache` parameter into the `Options` class. High-dimensional Expectation-Maximization (EM) operations are serialized, allowing rapid iteration on visualization and reporting without repeatedly waiting on matrix re-calculations.
*   **Numba JIT Acceleration:** The core Expectation-Maximization algorithm and Kalman Filter transition loop have been rewritten for ahead-of-time C compilation via `@numba.jit`. This drastically cuts execution time for extensive parameter searches (`max_iter` 5000+).
*   **Automated Econometric Validation:** Integrated Augmented Dickey-Fuller (ADF) testing via `statsmodels` to evaluate series stationarity post-transformation, directly guarding against feeding non-stationary data into the DFM.
*   **Testing Suite:** A comprehensive `pytest` suite enforces the mathematical integrity of the Numba translation against the pure Python implementation, particularly concerning NaN propagation ("ragged edges") inherent in raw macroeconomic releases.
*   **Dependency Management:** Packaged with `pyproject.toml` and a `hatchling` build backend; supports Python 3.10+.
*   **HTML Reporting:** Automatically generates standalone Plotly HTML reports containing Factor Contributions, Likelihood optimizations, and Model loadings.
*   **"News" Attribution Waterfalls:** Provides native bindings (`sp_news.py`) to dissect the mathematical drivers behind week-over-week DFM forecast changes. Generates Plotly Waterfall charts bridging `Actual vs Expected` impacts weighted by the Kalman gain.
*   **Pseudo-Real-Time Synthesizer:** Includes `sp_vintage_generator.py` for automatically simulating historical "ragged-edge" data matrices from a single modern dataset. Maps execution algorithms (e.g. `1st Friday of the Month`, `15th of the Month`) to dynamically blind data that had not yet been published, allowing mathematically un-cheated backtesting independent of the FRED ALFRED API.

## Repository Structure

* `data/` : Example US macro series retrieved from [FRED](https://fred.stlouisfed.org/).
* `dfm_sp/` : Core package encompassing the execution engines and analytics.
    * `core/` : Kalman filtering, spline imputation, EM-step mathematics, and data loaders.
    * `tests/` : Component tests simulating missing economic data and ragged tails.
* `examples/` : Ready-to-run scripts and configuration examples (CSV pipeline, custom config, spec generation).
* `notebooks/` : Jupyter notebooks covering classic usage, news waterfalls, pseudo-vintages, and weekly integration.
* `main.py` : Execution script illustrating the estimation of a standard panel and generating visualization artifacts.
* `example_Nowcast.py` : Demonstration of out-of-sample prediction mechanics (e.g., real GDP growth).

## Attribution & Notice

This package is not affiliated with the Federal Reserve Bank of New York.

The lineage of this work is:

1. **Eric Qian & Brandyn Bok (FRBNY)** — original MATLAB implementation of the DFM nowcasting framework ([FRBNY-TimeSeriesAnalysis/Nowcasting](https://github.com/FRBNY-TimeSeriesAnalysis/Nowcasting))
2. **Galib Khan (MajesticKhan)** — independent Python translation ([MajesticKhan/Nowcasting-Python](https://github.com/MajesticKhan/Nowcasting-Python))
3. **Sermet Pekin** — this package: modernised API, caching, plotting layer, news decomposition, vintage synthesizer, and testing suite

Academic credit for the methodology: Bok, Caratelli, Giannone, Sbordone & Tambalotti, "[Macroeconomic Nowcasting and Forecasting with Big Data](https://www.newyorkfed.org/research/staff_reports/sr830.html)", *Staff Reports 830*, Federal Reserve Bank of New York.
