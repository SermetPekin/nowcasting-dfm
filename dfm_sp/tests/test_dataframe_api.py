"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

Tests for the DataFrame-based API:
  - load_data_pandas   : accepts a DataFrame instead of a file path
  - run_with_dataframe : end-to-end DFM run with spec file + DataFrame
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from dfm_sp.core.load_data_pandas import load_data_pandas
from dfm_sp.sp_run import run_with_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPEC_FILE = PROJECT_ROOT / "Spec_US_example.xls"
DATA_FILE = PROJECT_ROOT / "data" / "US" / "2016-06-29.xls"

_has_sample_data = SPEC_FILE.exists() and DATA_FILE.exists()
needs_sample_data = pytest.mark.skipif(
    not _has_sample_data,
    reason="Sample data not present — run download_sample_data() first.",
)


# ---------------------------------------------------------------------------
# Minimal spec stub — avoids loading real files for pure unit tests
# ---------------------------------------------------------------------------


class _StubSpec:
    """Minimal spec object accepted by load_data_pandas."""

    def __init__(self, series_ids, transformations=None, frequencies=None):
        n = len(series_ids)
        self.SeriesID = np.array(series_ids)
        self.SeriesName = np.array(series_ids)
        self.Transformation = np.array(transformations or ["lin"] * n)
        self.Frequency = np.array(frequencies or ["m"] * n)


def _make_synthetic_df(n_series=3, n_periods=24, series_ids=None):
    """Build a simple monthly DataFrame with known values."""
    if series_ids is None:
        series_ids = [f"S{i}" for i in range(n_series)]
    dates = pd.date_range("2000-01-01", periods=n_periods, freq="MS")
    data = {
        sid: np.arange(1, n_periods + 1, dtype=float) * (i + 1)
        for i, sid in enumerate(series_ids)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    df = df.reset_index()  # move DatetimeIndex → "Date" column
    return df, series_ids


# ===========================================================================
# Unit tests — load_data_pandas (no real files needed)
# ===========================================================================


def test_load_data_pandas_output_shapes():
    """X, Time, Z must have consistent shapes after load_data_pandas."""
    df, series_ids = _make_synthetic_df(n_series=3, n_periods=24)
    spec = _StubSpec(series_ids)

    X, Time, Z = load_data_pandas(df, spec, date_col="Date")

    assert X.ndim == 2, "X must be 2-D"
    assert Z.ndim == 2, "Z must be 2-D"
    assert X.shape[1] == len(series_ids), "X columns must equal number of series"
    assert Z.shape[1] == len(series_ids), "Z columns must equal number of series"
    assert X.shape[0] == len(Time), "X rows must match Time length"
    assert Z.shape[0] == len(Time), "Z rows must match Time length"


def test_load_data_pandas_lin_transformation_passthrough():
    """With 'lin' transformation, Z and X values must be identical (no transform)."""
    df, series_ids = _make_synthetic_df(n_series=2, n_periods=24)
    spec = _StubSpec(series_ids, transformations=["lin", "lin"])

    X, Time, Z = load_data_pandas(df, spec, date_col="Date")

    # 'lin' keeps values unchanged — X and Z should be numerically equal
    mask = ~np.isnan(X) & ~np.isnan(Z)
    np.testing.assert_allclose(X[mask], Z[mask], rtol=1e-9)


def test_load_data_pandas_missing_date_col_raises():
    """Passing a DataFrame without the expected date column must raise ValueError."""
    df, series_ids = _make_synthetic_df()
    spec = _StubSpec(series_ids)

    df_no_date = df.rename(columns={"Date": "timestamp"})

    with pytest.raises(ValueError, match="Date"):
        load_data_pandas(df_no_date, spec, date_col="Date")


def test_load_data_pandas_custom_date_col():
    """date_col parameter correctly identifies an alternative date column name."""
    df, series_ids = _make_synthetic_df()
    spec = _StubSpec(series_ids)

    df_renamed = df.rename(columns={"Date": "observation_date"})

    X, Time, Z = load_data_pandas(df_renamed, spec, date_col="observation_date")
    assert len(Time) > 0


def test_load_data_pandas_sample_truncation():
    """The sample parameter (single start date) restricts output rows."""
    df, series_ids = _make_synthetic_df(n_periods=36)
    spec = _StubSpec(series_ids)

    X_full, Time_full, _ = load_data_pandas(df, spec, date_col="Date")
    # dropData accepts a single cutoff — rows with Time >= sample_start are kept
    X_trunc, Time_trunc, _ = load_data_pandas(
        df, spec, date_col="Date", sample="2001-06-01"
    )

    assert len(Time_trunc) < len(Time_full), "Truncated sample must be shorter"
    assert len(Time_trunc) > 0, "Truncated sample must not be empty"


def test_load_data_pandas_column_ordering():
    """Series columns are reordered to match spec order, not DataFrame column order."""
    series_ids = ["C", "A", "B"]
    df, _ = _make_synthetic_df(series_ids=series_ids)
    # Shuffle columns in the DataFrame
    df_shuffled = df[["Date", "B", "C", "A"]]

    spec = _StubSpec(series_ids)  # spec order: C, A, B
    X, Time, Z = load_data_pandas(df_shuffled, spec, date_col="Date")

    assert X.shape[1] == 3


# ===========================================================================
# Integration tests — run_with_dataframe (require sample data files)
# ===========================================================================


@needs_sample_data
def _load_vintage_as_dataframe():
    """Helper: load the sample vintage Excel file as a DataFrame."""
    return pd.read_excel(DATA_FILE)


@needs_sample_data
def test_run_with_dataframe_result_keys():
    """run_with_dataframe returns a ResultObject with expected DFM keys."""
    from dfm_sp.core.load_spec import LoadSpec
    from dfm_sp import Options

    df = _load_vintage_as_dataframe()
    Spec = LoadSpec(str(SPEC_FILE))
    options = Options(
        root=PROJECT_ROOT,
        spec_file_name=str(SPEC_FILE),
        max_iter=10,
        threshold=1e-3,
        use_cache=False,
        use_numba=False,
    )

    result_obj = run_with_dataframe(
        df, Spec, run_options=options, sample_start="2000-01-01", verbose=False
    )

    for key in ("x_sm", "X_sm", "Z", "C", "R", "A", "Q", "loglik"):
        assert key in result_obj.result, f"Missing key '{key}' in DFM result"


@needs_sample_data
def test_run_with_dataframe_matches_file_pipeline(pipeline_result):
    """DataFrame path with explicit sample= produces same X as the file-based pipeline."""
    from dfm_sp.core.load_data_pandas import load_data_pandas

    options, Spec, X_ref, Time_ref, Z_ref, _ = pipeline_result

    df = _load_vintage_as_dataframe()
    # Explicitly pass the same sample_start — without it, no truncation is applied
    # (which is the correct default for the DataFrame API).
    X_df, Time_df, Z_df = load_data_pandas(
        df, Spec, date_col="Date", sample=options.sample_start
    )

    assert (
        X_df.shape == X_ref.shape
    ), "Shape mismatch between DataFrame and file pipeline"

    # Values must agree where both are non-NaN
    mask = ~np.isnan(X_df) & ~np.isnan(X_ref)
    np.testing.assert_allclose(
        X_df[mask],
        X_ref[mask],
        rtol=1e-9,
        err_msg="DataFrame and file-based pipelines produced different transformed data",
    )


@needs_sample_data
def test_run_with_dataframe_loglik_finite():
    """Log-likelihood values from run_with_dataframe must all be finite."""
    from dfm_sp.core.load_spec import LoadSpec
    from dfm_sp import Options

    df = _load_vintage_as_dataframe()
    Spec = LoadSpec(str(SPEC_FILE))
    options = Options(
        root=PROJECT_ROOT,
        spec_file_name=str(SPEC_FILE),
        max_iter=10,
        threshold=1e-3,
        use_cache=False,
        use_numba=False,
    )

    result_obj = run_with_dataframe(
        df, Spec, run_options=options, sample_start="2000-01-01", verbose=False
    )
    loglik = result_obj.result["loglik"]

    assert len(loglik) > 1
    assert all(
        np.isfinite(v) for v in loglik[1:]
    ), "Non-finite log-likelihood — numerical instability in EM via DataFrame path"
