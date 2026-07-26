import pandas as pd
import numpy as np

from dfm_sp.core.load_data import sortData, transformData, dropData


def load_data_pandas(df: pd.DataFrame, Spec, sample=None, date_col: str = "Date"):
    """
    Directly process a pandas DataFrame containing historical Macro data using a Spec configuration
    without needing an intermediate file (Excel).

    Arguments:
        df       - pd.DataFrame containing dates and values.
        Spec     - the loaded spec configuration from `load_spec.py`
        sample   - (optional) specific start and end points for sample
        date_col - (optional) string representing the column name holding Observation Dates.

    Returns:
        X    : T x N numeric array, transformed dataset
        Time : T x 1 numpy array of dates
        Z    : T x N numeric array, raw data
    """

    # Store Date array
    if date_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{date_col}' column.")

    Time = df[date_col].to_numpy(copy=True)

    # Isolate Series Data (Mnem matching the matlab term for series names)
    Mnem = np.array([col for col in df.columns if col != date_col])
    Z = df[Mnem].to_numpy(copy=True)

    # Sort data variables based purely on what's defined in the Spec object
    Z, _ = sortData(Z.copy(), Mnem.copy(), Spec)

    # Apply statistical transformation (logs, differences, per-cents)
    X, Time, Z = transformData(Z.copy(), Time.copy(), Spec)

    # Truncate
    if sample is not None:
        X, Time, Z = dropData(X.copy(), Time.copy(), Z.copy(), sample)

    return X, Time, Z
