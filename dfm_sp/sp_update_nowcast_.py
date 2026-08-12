"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import numpy as np

from dfm_sp.core.update_Nowcast import update_nowcast
from dfm_sp.sp_classes import Options
from dfm_sp.sp_run import run, get_with_options
from dfm_sp.sp_news import plot_news_waterfall
from dfm_sp.sp_cache import CacheHandler

CACHE_DIR = ".pickles"
cache_ = CacheHandler(cache_dir=CACHE_DIR)
cache_.enable_persistent(True)


def sp_update_nowcast(options, new_date: str, series: str, period: str, show=True):

    vintage_old = options.vintage_date
    vintage_new = new_date

    options_baseline = options

    options_new = options_baseline.copy()
    options_new.vintage = new_date

    # Use the proper API
    print(f"Executing Baseline for {vintage_old}...")
    Spec, X_old, _Time_old, _Z_old = get_with_options(options_baseline)
    ResObject_base = run(X_old, Spec, options_baseline)
    Res_base = ResObject_base.result

    print(f"Loading New Data Vintage for {vintage_new}...")
    Spec_new, X_new, Time, _Z_new = get_with_options(options_new)

    y_old, y_new, news_table, data_released = update_nowcast(
        X_old, X_new, Time, Spec, Res_base, series, period, vintage_old, vintage_new
    )

    if not np.any(data_released):
        raise ValueError("No new Data! [Line 45]")
            
    real_impacts = news_table.iloc[np.where(data_released)[0], :]
    print(real_impacts)
    if real_impacts.empty : 
        raise ValueError(" No forecast was made ")
        
    fig = plot_news_waterfall(
        news_table=real_impacts,
        y_old=y_old,
        y_new=y_new,
        vintage_old=options.vintage_date,
        vintage_new=options_new.vintage_date,
        target_series=f"{series} - {str(period).upper() }",
    )
    if show:
        fig.show()

    return {
        "fig": fig,
        "real_impacts": real_impacts,
        "data_released": data_released,
        "news_table": news_table,
    }


def _usage():
    series = "GDPC1"
    period = "2016q4"
    vintage_old = "2016-12-16"
    vintage_new = "2016-12-23"
    sample_start = "2005-01-01"
    country = "US"
    spec_file_name = "Spec_US_example.xls"
    max_iter = 5000

    options_baseline = Options(
        vintage=vintage_old,
        country=country,
        spec_file_name=spec_file_name,
        max_iter=max_iter,
        use_cache=True,
        sample_start=sample_start,
    )

    result_dict = sp_update_nowcast(options_baseline, vintage_new, series, period)
    print(result_dict.keys())
