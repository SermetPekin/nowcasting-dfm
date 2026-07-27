"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import os
from datetime import datetime as dt
import pickle
import sys
from pathlib import Path

# -------------------------------------------------Libraries
# Libs
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# ================================================================================
# DFM Spkn edition
from dfm_sp.core.load_spec import LoadSpec
from dfm_sp.core.load_data import load_data
from dfm_sp.core.dfm import dfm
from dfm_sp.core.summarize import summarize
from dfm_sp.sp_utils import get_latest, Timer
from dfm_sp.sp_plots import plot_transformed_data
from dfm_sp.sp_plots import plot_loglik, plot_projection_x_over_y
from dfm_sp.sp_classes import Options, ResultObject


@Timer()
def run(X, Spec, run_options: Options = None) -> ResultObject:
    # Run dynamic factor model (DFM) and save estimation output as 'ResDFM'.
    if run_options is None:
        run_options = Options()
    Spec.run_options = run_options
    hash_ = run_options.hash()
    OUT_FILE = (
        Path(".pickles")
        / f"ResDFM-iter_V-{run_options.vintage}-M-{run_options.max_iter}_T-{run_options.threshold}-hash-{hash_}.pickle"
    )
    Path(".pickles").mkdir(exist_ok=True)
    print(f"creating file {OUT_FILE} ")
    if OUT_FILE.exists() and run_options.use_cache:
        print("[CACHE FOUND] Loading previous results from cache...")
        with open(str(OUT_FILE), "rb") as f:
            return pickle.load(f)
    elif OUT_FILE.exists():
        print(
            f"[CACHE IGNORED] run_options.use_cache is False. Re-running calculation."
        )
    run_options.check()
    Res = dfm(
        X,
        Spec,
        run_options.threshold,
        max_iter=run_options.max_iter,
        use_numba=run_options.use_numba,
    )
    res_object = ResultObject(Res, Spec, run_options)
    with open(str(OUT_FILE), "wb") as handle:
        pickle.dump(res_object, handle)
    # TODO: Res and Spec should be separate, this will be fixed after the unit tests are created
    # [sp] Added some unit tests
    return res_object


def get_with_options(options: Options, verbose = True):
    Spec : LoadSpec = LoadSpec(options.spec_file_name)
    datafile = options.root / os.path.join(
        "data", options.country, options.vintage_date + ".xls"
    )
    if not datafile.exists():
        raise FileNotFoundError(str(datafile))
    X, Time, Z = load_data(datafile, Spec, options.sample_start)
    if verbose :        
        summarize(X, Time, Spec)
    return Spec, X, Time, Z

def run_with_options(options: Options, verbose = True) -> ResultObject:
    Spec, X, Time, Z = get_with_options(options, verbose)
    ResObject: ResultObject = run(X, Spec, options)
    return ResObject

def plot_with_options(options: Options):
    Spec, X, Time, Z = get_with_options(options)
    plot_transformed_data(Spec, X, Z, Time, options.plot1_series)
    ResObject: ResultObject = run(X, Spec, options)
    plot_loglik(ResObject, Time)
    for items in options.plot2_series:
        plot_projection_x_over_y(ResObject, X, Z, Time, items)
