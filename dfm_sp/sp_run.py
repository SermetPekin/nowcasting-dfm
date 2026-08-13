"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

# -------------------------------------------------Libraries
# Libs
import numpy as np
import pandas as pd

# ================================================================================

from dfm_sp.core.load_spec import LoadSpec
from dfm_sp.core.load_data import load_data
from dfm_sp.core.load_data_pandas import load_data_pandas
from dfm_sp.core.dfm import dfm
from dfm_sp.core.summarize import summarize
from dfm_sp.sp_utils import Timer
from dfm_sp.sp_plots import plot_transformed_data
from dfm_sp.sp_plots import plot_loglik, plot_projection_x_over_y
from dfm_sp.sp_classes import Options, ResultObject


@Timer()
def run(
    X: np.ndarray,
    Spec: LoadSpec,
    run_options: Optional[Options] = None,
    verbose: Optional[bool] = None,
) -> ResultObject:
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
        print("[CACHE IGNORED] run_options.use_cache is False. Re-running calculation.")
    run_options.check()
    Res = dfm(
        X,
        Spec,
        run_options.threshold,
        max_iter=run_options.max_iter,
        use_numba=run_options.use_numba,
        verbose=run_options.get_verbose(verbose),
    )
    res_object = ResultObject(Res, Spec, run_options)
    with open(str(OUT_FILE), "wb") as handle:
        pickle.dump(res_object, handle)
    # TODO: Res and Spec should be separate, this will be fixed after the unit tests are created
    # [sp] Added some unit tests
    return res_object


def get_with_options(
    options: Options, verbose: Optional[bool] = None
) -> Tuple[LoadSpec, np.ndarray, np.ndarray, np.ndarray]:

    verbose = options.get_verbose(verbose)

    Spec: LoadSpec = LoadSpec(options.spec_file_name)
    
    if callable(options.data_file_name_format) : 
        datafile : Path  = options.data_folder / options.data_file_name_format(options) 
    else : 
        datafile : Path = options.data_folder / (options.vintage_date + ".xls" )   
    if not datafile.suffix : 
        datafile = datafile.with_suffix(".xls")  
    if not datafile.exists():
        raise FileNotFoundError(str(datafile))
    X, Time, Z = load_data(datafile, Spec, options.sample_start)
    if verbose:
        summarize(X, Time, Spec)
    return Spec, X, Time, Z


def run_with_options(options: Options, verbose: Optional[bool] = None) -> ResultObject:
    verbose = options.get_verbose(verbose)
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


def run_with_dataframe(
    df: pd.DataFrame,
    Spec: LoadSpec,
    run_options: Optional[Options] = None,
    sample_start: Optional[str] = None,
    date_col: str = "Date",
    verbose: Optional[bool] = None,
) -> ResultObject:
    """Run DFM using a pandas DataFrame as the data source instead of a vintage file.

    The spec file (LoadSpec) still defines the model structure — series IDs,
    frequencies, transformations, and block loadings — exactly as before.
    Only the data source changes: a pre-loaded DataFrame replaces the Excel/CSV file.

    Arguments:
        df           - DataFrame with a date column and one column per series.
                       Column names must match the SeriesIDs in the spec.
        Spec         - Model specification loaded via LoadSpec (from Excel or SpecConfig).
        run_options  - Optional Options object controlling EM iterations, caching, etc.
                       Note: run_options.sample_start is intentionally ignored here.
                       Use the ``sample_start`` parameter instead if truncation is needed.
        sample_start - Optional start date string (e.g. "2010-01-01"). Rows before this
                       date are dropped before estimation. Defaults to None — no
                       truncation is applied, which is the correct default when the
                       caller controls the DataFrame date range.
        date_col     - Name of the date column in df (default: "Date").
        verbose      - Override verbosity; falls back to run_options.verbose if None.

    Returns:
        ResultObject with DFM estimation results.

    Example::

        import pandas as pd
        from dfm_sp import LoadSpec, run_with_dataframe

        df = pd.read_csv("my_data.csv")          # caller controls date range
        Spec = LoadSpec("Spec_US_example.xls")   # existing spec file unchanged
        result = run_with_dataframe(df, Spec)
        # or, to drop rows before 2010:
        result = run_with_dataframe(df, Spec, sample_start="2010-01-01")
    """
    if run_options is None:
        run_options = Options()
    # run_options.sample_start is deliberately NOT used here — the caller owns
    # the DataFrame date range. sample_start parameter controls truncation explicitly.
    X, Time, Z = load_data_pandas(df, Spec, sample=sample_start, date_col=date_col)
    if run_options.get_verbose(verbose):
        summarize(X, Time, Spec)
    return run(X, Spec, run_options, verbose)
