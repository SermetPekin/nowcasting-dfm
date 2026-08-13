"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

from pathlib import Path

from dfm_sp.sp_plots import (
    plot_transformed_data,
    plot_loglik,
    plot_common,
    plot_projection_x_over_y,
)
from dfm_sp.sp_classes import Options, ResultObject
from dfm_sp.sp_run import plot_with_options, run, get_with_options
from dfm_sp.sp_plots import (
    plot_factor_contribution,
    plot_factors_with_series,
    plot_prediction_intervals,
)
from dfm_sp.sp_heatmap import plot_factor_loadings
from dfm_sp.sp_plot_generator import generate_html_report


def daily_report(options: Options | None = None, write = True):
    ROOT = Path(".")

    if options is None:
        options = Options(ROOT)
    Spec, X, Time, Z = get_with_options(options)
    ResObject: ResultObject = run(X, Spec, options)
    generate_html_report(ResObject, Time, X, Z, options)
    if write : 
        ResObject.write()
    return ResObject 


def main_interactive():
    SHOW = True
    ROOT = Path(".")

    options = Options(ROOT)
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
