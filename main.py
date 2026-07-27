from pathlib import Path

# -------------------------------------------------Libraries
# Libs
import pandas as pd

# ================================================================================
# DFM Spkn edition
from dfm_sp import (
    Options,
    get_with_options,
    plot_with_options,
    plot_transformed_data,
    ResultObject,
    plot_loglik,
    plot_common,
    run,
    plot_projection_x_over_y,
    plot_prediction_intervals,
)
from dfm_sp.sp_plots import (
    plot_factor_contribution,
    plot_factors_with_series,
    plot_prediction_intervals,
)
from dfm_sp import plot_factor_loadings
from dfm_sp import generate_html_report


# ================================================================================
def main():
    Spec, X, Time, Z = get_with_options(options)
    ResObject: ResultObject = run(X, Spec, options)
    generate_html_report(ResObject, Time, X, Z, options)


def main_interactive():
    SHOW = True
    Spec, X, Time, Z = get_with_options(options)
    ResObject: ResultObject = run(X, Spec, options)
    plot_with_options(options)
    plot_transformed_data(Spec, X, Z, Time, options.plot1_series)
    plot_loglik(ResObject, Time, show=SHOW)
    plot_common(ResObject, Time, show=SHOW)
    plot_prediction_intervals(ResObject, Time, show=SHOW)
    plot_factor_contribution(ResObject, show=SHOW)
    plot_factor_loadings(ResObject, show=SHOW)
    plot_factors_with_series(ResObject, Time, options.plot1_series, show=SHOW)
    for items in options.plot2_series:
        plot_projection_x_over_y(ResObject, X, Z, Time, items, show=SHOW)


ROOT = Path(".")
options = Options(ROOT, max_iter=5000, use_cache=False)
main()
# main_interactive()
