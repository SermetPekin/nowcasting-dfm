"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import numpy as np
import pandas as pd 

from dfm_sp.core.update_Nowcast import update_nowcast
from dfm_sp.sp_classes import Options
from dfm_sp.sp_run import run, get_with_options
from dfm_sp.sp_news import plot_news_waterfall
from dfm_sp.sp_cache import CacheHandler

CACHE_DIR = ".pickles"
cache_ = CacheHandler(cache_dir=CACHE_DIR)
cache_.enable_persistent(True)


def sp_update_nowcast(options, new_date: str, series: str, period: str, show=True, write=True):

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
            
    real_impacts : pd.DataFrame = news_table.iloc[np.where(data_released)[0], :]
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

    result_dict =  {
        "fig": fig,
        "real_impacts": real_impacts,
        "data_released": data_released,
        "news_table": news_table,
    }
    if write : 
        write_result_dict(result_dict, "out_impacts")
    return result_dict

def write_result_dict(result_dict:dict, file_name = "out_impacts"):
    """write outputs of update nowcast """
    data_released = pd.DataFrame(result_dict["data_released"])
    news_table = result_dict["news_table"]
    real_impacts = result_dict["real_impacts"] 
    real_impacts['Absolute Impact'] = real_impacts['Impact'].abs()   
    real_impacts = real_impacts.sort_values(by='Absolute Impact', ascending=False)
    
    with pd.ExcelWriter(f"{file_name}.xlsx", engine="openpyxl") as writer:
        data_released.to_excel(writer, sheet_name="Data Released", index=True)
        news_table.to_excel(writer, sheet_name="News Table", index=True)
        real_impacts.to_excel(writer, sheet_name="Real Impacts", index=True)

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
