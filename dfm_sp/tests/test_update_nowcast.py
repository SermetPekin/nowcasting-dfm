"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

Integration tests for sp_update_nowcast and write_result_dict.

Uses the Dec-2016 vintage pair (2016-12-16 → 2016-12-23) which is the
canonical example from the FRBNY replication kit and is always present
in the sample dataset.  All tests are skipped when the data files are absent.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPEC_FILE = PROJECT_ROOT / "Spec_US_example.xls"
VINTAGE_OLD = "2016-12-16"
VINTAGE_NEW = "2016-12-23"
DATA_OLD = PROJECT_ROOT / "data" / "US" / f"{VINTAGE_OLD}.xls"
DATA_NEW = PROJECT_ROOT / "data" / "US" / f"{VINTAGE_NEW}.xls"

_has_data = SPEC_FILE.exists() and DATA_OLD.exists() and DATA_NEW.exists()
needs_data = pytest.mark.skipif(
    not _has_data,
    reason="Sample data not present — run download_sample_data() first.",
)

# Target series / period used by the FRBNY replication example
TARGET_SERIES = "GDPC1"
TARGET_PERIOD = "2016q4"


# ---------------------------------------------------------------------------
# Session fixture — run sp_update_nowcast once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def update_result():
    """Run sp_update_nowcast with the Dec-2016 vintage pair (max_iter=10)."""
    if not _has_data:
        pytest.skip("Sample data not present.")

    from dfm_sp import Options
    from dfm_sp.sp_update_nowcast_ import sp_update_nowcast

    options = Options(
        root=PROJECT_ROOT,
        vintage=VINTAGE_OLD,
        country="US",
        spec_file_name=str(SPEC_FILE),
        sample_start="2005-01-01",
        max_iter=10,
        threshold=1e-3,
        use_cache=False,
        use_numba=False,
        verbose=False,
    )

    result = sp_update_nowcast(
        options,
        new_date=VINTAGE_NEW,
        series=TARGET_SERIES,
        period=TARGET_PERIOD,
        show=False,
        write=False,
    )
    return result


# ---------------------------------------------------------------------------
# Tests for result_dict structure
# ---------------------------------------------------------------------------


@needs_data
def test_result_dict_has_required_keys(update_result):
    """sp_update_nowcast must return a dict with all documented keys."""
    assert isinstance(update_result, dict)
    for key in ("fig", "real_impacts", "data_released", "news_table"):
        assert key in update_result, f"Missing key '{key}' in result_dict"


@needs_data
def test_news_table_is_dataframe(update_result):
    """news_table must be a pandas DataFrame."""
    assert isinstance(update_result["news_table"], pd.DataFrame)


@needs_data
def test_news_table_has_required_columns(update_result):
    """news_table must have the four standard impact-decomposition columns."""
    news = update_result["news_table"]
    for col in ("Forecast", "Actual", "Weight", "Impact"):
        assert col in news.columns, f"Missing column '{col}' in news_table"


@needs_data
def test_news_table_indexed_by_series_id(update_result):
    """news_table index must contain recognisable macro series identifiers."""
    news = update_result["news_table"]
    assert len(news.index) > 0, "news_table must have at least one row"
    # All index values should be non-empty strings
    assert all(isinstance(s, str) and len(s) > 0 for s in news.index)


@needs_data
def test_news_table_numeric_values(update_result):
    """Forecast, Actual, Weight, and Impact columns must all be numeric."""
    news = update_result["news_table"]
    for col in ("Forecast", "Actual", "Weight", "Impact"):
        assert pd.api.types.is_numeric_dtype(
            news[col]
        ), f"Column '{col}' is not numeric"


@needs_data
def test_news_table_impact_finite(update_result):
    """Impact values must all be finite (no NaN / Inf)."""
    impacts = update_result["news_table"]["Impact"].dropna()
    assert len(impacts) > 0
    assert np.all(
        np.isfinite(impacts.values)
    ), "Non-finite Impact values found in news_table"


# ---------------------------------------------------------------------------
# Tests for data_released
# ---------------------------------------------------------------------------


@needs_data
def test_data_released_is_boolean_array(update_result):
    """data_released must be a boolean numpy array."""
    dr = update_result["data_released"]
    assert isinstance(dr, np.ndarray), "data_released must be a numpy array"
    assert dr.dtype == bool, f"data_released dtype should be bool, got {dr.dtype}"


@needs_data
def test_data_released_has_at_least_one_release(update_result):
    """Between the two Dec-2016 vintages at least one series must have been released."""
    assert update_result["data_released"].any(), (
        "No data releases detected between the two vintages — "
        "check that VINTAGE_OLD and VINTAGE_NEW are different"
    )


# ---------------------------------------------------------------------------
# Tests for real_impacts (subset of news_table)
# ---------------------------------------------------------------------------


@needs_data
def test_real_impacts_is_subset_of_news_table(update_result):
    """real_impacts rows must all appear in news_table."""
    news = update_result["news_table"]
    real = update_result["real_impacts"]
    assert isinstance(real, pd.DataFrame)
    assert set(real.index).issubset(
        set(news.index)
    ), "real_impacts contains series not present in news_table"


@needs_data
def test_real_impacts_size_matches_data_released(update_result):
    """Number of real_impacts rows must equal number of released series."""
    n_released = update_result["data_released"].sum()
    n_real = len(update_result["real_impacts"])
    assert (
        n_real == n_released
    ), f"real_impacts has {n_real} rows but {n_released} series were released"


# ---------------------------------------------------------------------------
# Test for write_result_dict
# ---------------------------------------------------------------------------


@needs_data
def test_write_result_dict_creates_excel_with_sheets(update_result, tmp_path):
    """write_result_dict must produce an Excel file with the three expected sheets."""
    from dfm_sp.sp_update_nowcast_ import write_result_dict

    out_file = tmp_path / "test_out_impacts"
    write_result_dict(update_result, file_name=str(out_file))

    xlsx_path = Path(str(out_file) + ".xlsx")
    assert xlsx_path.exists(), "Excel output file was not created"

    xl = pd.ExcelFile(str(xlsx_path))
    for sheet in ("Data Released", "News Table", "Real Impacts"):
        assert sheet in xl.sheet_names, f"Sheet '{sheet}' missing from Excel output"


@needs_data
def test_write_result_dict_real_impacts_sorted_by_absolute_impact(
    update_result, tmp_path
):
    """Real Impacts sheet must be sorted descending by absolute impact."""
    from dfm_sp.sp_update_nowcast_ import write_result_dict

    out_file = tmp_path / "test_sorted"
    write_result_dict(update_result, file_name=str(out_file))

    df = pd.read_excel(str(out_file) + ".xlsx", sheet_name="Real Impacts", index_col=0)
    if len(df) > 1:
        abs_impacts = df["Impact"].abs().values
        assert np.all(
            abs_impacts[:-1] >= abs_impacts[1:]
        ), "Real Impacts sheet is not sorted by absolute impact (descending)"


# ---------------------------------------------------------------------------
# Column order invariance
# ---------------------------------------------------------------------------


@needs_data
def test_column_order_does_not_affect_transformed_data():
    """Shuffling data columns must produce an identical transformed matrix.

    sortData() inside load_data_pandas reorders columns to match the spec,
    so the order in which columns arrive should have no effect on X, Time, Z.
    """
    from dfm_sp.core.load_spec import LoadSpec
    from dfm_sp.core.load_data_pandas import load_data_pandas

    Spec = LoadSpec(str(SPEC_FILE))
    df = pd.read_excel(str(DATA_OLD))

    # Canonical order
    X_ref, Time_ref, Z_ref = load_data_pandas(df.copy(), Spec, date_col="Date")

    # Shuffle all non-Date columns with a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    series_cols = [c for c in df.columns if c != "Date"]
    shuffled_cols = rng.permutation(series_cols).tolist()
    df_shuffled = df[["Date"] + shuffled_cols]

    X_shuf, Time_shuf, Z_shuf = load_data_pandas(df_shuffled, Spec, date_col="Date")

    assert X_ref.shape == X_shuf.shape, "Shape changed after column shuffle"

    # Compare where both are non-NaN
    mask = ~np.isnan(X_ref) & ~np.isnan(X_shuf)
    np.testing.assert_allclose(
        X_ref[mask],
        X_shuf[mask],
        rtol=1e-12,
        err_msg="Transformed data differs when input column order is shuffled",
    )


@needs_data
def test_reversed_column_order_does_not_affect_transformed_data():
    """Reversing the column order must also produce identical results."""
    from dfm_sp.core.load_spec import LoadSpec
    from dfm_sp.core.load_data_pandas import load_data_pandas

    Spec = LoadSpec(str(SPEC_FILE))
    df = pd.read_excel(str(DATA_OLD))

    X_ref, _, _ = load_data_pandas(df.copy(), Spec, date_col="Date")

    series_cols = [c for c in df.columns if c != "Date"]
    df_reversed = df[["Date"] + series_cols[::-1]]

    X_rev, _, _ = load_data_pandas(df_reversed, Spec, date_col="Date")

    mask = ~np.isnan(X_ref) & ~np.isnan(X_rev)
    np.testing.assert_allclose(
        X_ref[mask],
        X_rev[mask],
        rtol=1e-12,
        err_msg="Transformed data differs when input column order is reversed",
    )
