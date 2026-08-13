"""
Step 2 — Basic model run
Loads a vintage dataset, fits the DFM, summarises the data,
and produces a set of exploratory Plotly figures.
"""

import pandas as pd

from dfm_sp import Options, get_with_options, run, summarize
from dfm_sp.sp_plots import (
    plot_transformed_data_one,
    plot_loglik,
    plot_common,
    plot_projection_x_over_y,
)

# -------------------------------------------------User Inputs
options = Options(
    vintage="2016-06-29",
    country="US",
    sample_start="2000-01-01",
    spec_file_name="Spec_US_example.xls",
    threshold=1e-4,
    use_cache=False,
    max_iter=5000,
)

# Set dataframe to full view
pd.set_option("display.expand_frame_repr", False)

# -------------------------------------------------Load model specification and dataset.
Spec, X, Time, Z = get_with_options(options)

# Summarize dataset
summarize(X, Time, Spec)

# -------------------------------------------------Plot raw vs transformed data for a single series
fig_transformed = plot_transformed_data_one(
    Spec, X, Z, Time, series_id="INDPRO", show=False
)
fig_transformed.show()

# -------------------------------------------------Run dynamic factor model (DFM)
print("Running Dynamic Factor Model (DFM)...")
ResObject = run(X, Spec, options)

# -------------------------------------------------Plot LogLik convergence
fig_loglik = plot_loglik(ResObject, Time, show=False)
fig_loglik.show()

# -------------------------------------------------Plot common factor and standardized data
fig_common = plot_common(ResObject, Time, series_id="INDPRO", show=False)
fig_common.show()

# -------------------------------------------------Plot projection of common factor onto series of interest
fig_projection = plot_projection_x_over_y(
    ResObject, X, Z, Time, series=["PAYEMS", "GDPC1"], show=False
)
fig_projection.show()
