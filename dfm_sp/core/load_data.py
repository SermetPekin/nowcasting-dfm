"""
BSD 3-Clause License

Copyright (c) 2018, Federal Reserve Bank of New York (original MATLAB implementation by Eric Qian and Brandyn Bok)
Copyright (c) 2019, Galib Khan (independent Python translation, not affiliated with FRBNY, https://github.com/MajesticKhan/Nowcasting-Python)

"""

# -------------------------------------------------Libraries
import pandas as pd
import numpy as np
import os
import warnings
from typing import Tuple, Optional, Union
from pathlib import Path
from .load_spec import LoadSpec, SpecConfig


def check_stationarity(series: np.ndarray, name: str):
    """
    Check if a series is stationary using the Augmented Dickey-Fuller test.
    Only checks if the series has sufficient valid observations.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return  # statsmodels is not installed, skip test

    # Remove NaNs for the test
    valid_data = series[~np.isnan(series)]

    if len(valid_data) < 30:
        # Too few observations to reliably test
        return

    try:
        # Perform ADF test
        adf_result = adfuller(valid_data, autolag="AIC")
        p_value = adf_result[1]

        # If p-value > 0.05, we fail to reject the null hypothesis that the series has a unit root (is non-stationary)
        if p_value > 0.05:
            ...
            # warnings.warn(f"\n[STATIONARITY WARNING]: Series {name} might be non-stationary after transformation.\nADF p-value: {p_value:.4f} > 0.05. \nFeeding non-stationary data into a DFM will result in spurious unobserved factors.")
    except Exception as e:
        # Catch any exceptions during the test to prevent crashing the data load
        warnings.warn(f"Could not perform ADF test on {name}: {str(e)}")


# -------------------------------------------------Read data functions


def load_data(
    datafile: Union[str, Path],
    Spec: Union["LoadSpec", SpecConfig],
    sample: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load macro data against model specification from file and format as a scaled panel structure.

    Returns:
        X    : np.ndarray shape (T, N) - numeric array, statistically transformed dataset
        Time : np.ndarray shape (T,)   - 1D array of datetime64 or string dates
        Z    : np.ndarray shape (T, N) - numeric array, raw (untransformed) dataset
    """
    #

    """
    Python Version Notes:
        The original matlab function can load raw data from MATLAB formatted binary (.mat) file.
        However, in the Python version I have removed this feature.
        In transformData(), the formula_dict dictionary contains functions that transform the data. 
        I could have done it in a similar fashion as the Matlab code, however I find it easier to read the code when going through the if and elif statements
        When converting dates to ordinal format, we need to add 366 days to match with Matlab date numeric values
    
    [sp] dropped this ordinal format. now dates are kept as datetime64[ns] and passed to the DFM function as such.


    """
    if not os.path.splitext(datafile)[1].lower() in [".xlsx", ".xls", ".csv"]:
        raise ValueError("File is not an EXCEL or CSV FILE")
    Z, Time, Mnem = readData(datafile)
    #    Z : raw (untransformed) observed data
    # Time : observation periods for the time series data
    # Mnem : series ID for each variable
    # Sort data based on model specification
    Z, _ = sortData(Z.copy(), Mnem.copy(), Spec)
    # since now Mnem == Spec.SeriesID
    del Mnem
    # Transform data based on model specification
    X, Time, Z = transformData(Z.copy(), Time.copy(), Spec)
    # Drop data not in estimation sample
    if sample != None:
        X, Time, Z = dropData(X.copy(), Time.copy(), Z.copy(), sample)
    return X, Time, Z


def readData(datafile):
    if str(datafile).lower().endswith(".csv"):
        dat = pd.read_csv(datafile)
    else:
        dat = pd.read_excel(datafile)
    # Extract non-Date columns
    Mnem = np.array([col for col in dat.columns if col != "Date"])
    Z = dat[Mnem].to_numpy(copy=True)
    # Keep Time as datetime64 (no conversion needed)
    Time = dat["Date"].to_numpy(copy=True)  # or dat["Date"].values
    return Z, Time, Mnem


def sortData(Z, Mnem, Spec):
    # sortData Sort series by order of model specification
    # Drop series not in Spec
    # inSpec = np.in1d(Mnem, Spec.SeriesID)
    inSpec = np.isin(Mnem, Spec.SeriesID)  # [ 2026 fix by sp ]
    Mnem = Mnem[inSpec]
    Z = Z[:, inSpec]
    # Sort series by ordering of Spec
    permutation = np.array([np.where(Mnem == i)[0][0] for i in Spec.SeriesID])
    Mnem = Mnem[permutation]
    Z = Z[:, permutation]
    return Z, Mnem


def transformData(Z, Time, Spec):
    # transformData Transforms each data series based on Spec.Transformation
    #
    # Input Arguments:
    #
    #      Z : T x N numeric array, raw (untransformed) observed data
    #   Spec : structure          , model specification
    #
    # Output Arguments:
    #
    #      X : T x N numeric array, transformed data (stationary to enter DFM)
    """
    Transformation notes:

            lin = Levels (No Transformation)
            chg: Change (Difference)
            ch1: Year over Year Change (Difference)
            pch: Percent Change
            pc1: Year over Year Percent Change
            pca: Percent Change (Annual Rate)
            log: Natural Log

            --- added these -- [sp]

            cch: "Continuously Compounded Rate of Change
            cca: "Continuously Compounded Annual Rate of Change
            dln: Log Difference (First diff of log)
            dl1: Year over Year Log Difference
            d2l: Second Log Difference
            zsc: Z-Score Standardization

    """
    T, N = Z.shape
    X = np.empty((T, N))
    X[:] = np.nan
    Freq_dict = {"m": 1, "q": 3}

    from dfm_sp.sp_transformations import MacroTransformations

    # [sp] created a centralized registry for transformations to avoid hardcoding formulas in multiple places.

    cols = []
    for i in range(N):
        formula = Spec.Transformation[i]
        freq = Spec.Frequency[i]
        step = Freq_dict[
            freq
        ]  # time step for different frequencies based on monthly time
        t1 = (
            step - 1
        )  # assume monthly observations start at beginning of quarter (subtracted 1 for indexing)
        n = step / 12  # number of years, needed to compute annual % changes
        series_name = Spec.SeriesName[i]
        cols.append(series_name)

        Z_i = Z[:, i]

        # Pull formula definitions dynamically initialized with the array steps for this variable
        formula_dict = MacroTransformations.get_formulas(t1, step, n)
        transform = formula_dict.get(formula)

        if transform is None:
            raise ValueError(f"Unknown transformation: {formula}")

        # Standard Level / Logs apply across the whole array instantly
        if formula in {"lin", "log", "zsc"}:
            X[:, i] = transform(Z_i)

        # Standard First / 2nd / Percent Differencing require sliced targeting
        elif formula in {"chg", "pch", "pca", "dln", "d2l"}:
            X[t1::step, i] = transform(Z_i)

        # Annual Differencing skip 12 periods
        elif formula in {"ch1", "pc1", "dl1"}:
            X[12 + t1 :: step, i] = transform(Z_i)

        else:
            raise ValueError(f"Unsupported transformation: {formula}")

        # Check for stationarity after transformation
        check_stationarity(X[3:, i], series_name)

    # Drop first quarter of observations
    # since transformations cause missing values
    # Sp added this for Debugging
    d = pd.DataFrame(X[3:, :], columns=cols)
    d.to_excel(f"TransformedData.xlsx")
    return X[3:, :], Time[3:], Z[3:, :]


def dropData(X, Time, Z, sample):
    """Remove data not in estimation sample.
    Args:
        X: Feature matrix (n_samples, n_features)
        Time: Array of datetime64[ns] (n_samples,)
        Z: Target matrix (n_samples, n_targets)
        sample: Cutoff date (datetime64, str, or ordinal)
    Returns:
        Filtered X, Time, Z where Time >= sample
    """
    # Convert sample to datetime64 if it's a string or ordinal
    if isinstance(sample, str):
        sample = np.datetime64(sample)
    elif isinstance(sample, (int, np.integer)):
        sample = np.datetime64(sample, "D")  # Assume ordinal
    # Filter
    filter_index = Time >= sample
    X = X[filter_index, :].copy()
    Time = Time[filter_index].copy()
    Z = Z[filter_index, :].copy()
    return X, Time, Z
