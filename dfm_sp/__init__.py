import warnings

# Suppress annoying standard warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)

from dfm_sp.core.dfm import dfm
from dfm_sp.core.summarize import summarize
from dfm_sp.sp_utils import get_latest, Timer
from dfm_sp.sp_update_nowcast_ import sp_update_nowcast


from pathlib import Path
from dfm_sp.sp_daily import daily_report

# Exposed Public API (Easier names)
from dfm_sp.sp_classes import Options, ResultObject
from dfm_sp.core.load_spec import SpecConfig, load_spec
from dfm_sp.core.load_data import load_data
from dfm_sp.core.dfm import dfm
from dfm_sp.core.summarize import summarize
from dfm_sp.sp_run import run_with_options, run
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

__all__ = [
    # Core DFM Loop
    "Options",
    "SpecConfig",
    "ResultObject",
    "run_with_options",
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
    "load_spec",
    "load_data",
    "dfm",
    "summarize",
]
