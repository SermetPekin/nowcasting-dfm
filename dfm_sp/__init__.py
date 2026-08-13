from pathlib import Path

# Exposed Public API
from dfm_sp.sp_classes import Options, ResultObject
from dfm_sp.core.load_spec import (
    SpecConfig,
    LoadSpec,
    load_spec,
)  # load_spec is a backward-compat alias

from dfm_sp.core.update_Nowcast import update_nowcast

from dfm_sp.sp_daily import daily_report
from dfm_sp.sp_plots import (
    plot_factor_contribution,
    plot_factors_with_series,
    plot_prediction_intervals,
    plot_projection_x_over_y,
)
from dfm_sp.core.load_data import load_data
from dfm_sp.core.load_data_pandas import load_data_pandas
from dfm_sp.core.dfm import dfm
from dfm_sp.core.summarize import summarize
from dfm_sp.sp_run import get_with_options, run, run_with_options, run_with_dataframe
from dfm_sp.sp_update_nowcast_ import sp_update_nowcast
from dfm_sp.sp_plots import (
    plot_transformed_data,
    plot_loglik,
    plot_common,
    plot_loglik_together,
    plot_projection_x_over_y,
)
from dfm_sp.sp_plots_blocks import plot_block_contributions
from dfm_sp.sp_news import plot_news_waterfall

# Expose Vintage Synthesizer heavily simplified
from dfm_sp.sp_vintage_generator import VintageMaker, FixedDayRule, WeekdayRule

# Data download helper
from dfm_sp.sp_download import download_sample_data


import warnings

# Suppress annoying standard warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)


__all__ = [
    "Path",
    # Core DFM Loop
    "Options",
    "SpecConfig",
    "ResultObject",
    "get_with_options",
    "run",
    # Plotting/Visualizations
    "plot_transformed_data",
    "plot_loglik",
    "plot_common",
    "plot_loglik_together",
    "plot_projection_x_over_y",
    "plot_block_contributions",
    "plot_news_waterfall",
    # Pre-Processing
    "VintageMaker",
    "FixedDayRule",
    "WeekdayRule",
    # Classic Endpoints
    "LoadSpec",
    "load_spec",  # backward-compat alias for LoadSpec
    "load_data",
    "load_data_pandas",
    "dfm",
    "summarize",
    "daily_report",
    "download_sample_data",
    "run_with_options",
    "run_with_dataframe",
    "sp_update_nowcast",
]
