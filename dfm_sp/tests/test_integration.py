"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

Integration test: full pipeline run with real spec and vintage data.
Kept deliberately fast by capping max_iter at 10.
The pipeline_result fixture is provided by conftest.py (session-scoped).
"""

import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPEC_FILE = PROJECT_ROOT / "Spec_US_example.xls"
DATA_FILE = PROJECT_ROOT / "data" / "US" / "2016-06-29.xls"

pytestmark = pytest.mark.skipif(
    not SPEC_FILE.exists() or not DATA_FILE.exists(),
    reason="Sample data not present — run download_sample_data() first.",
)


def test_spec_loads(pipeline_result):
    """LoadSpec populates all required arrays from the real Excel spec."""
    _, Spec, _, _, _, _ = pipeline_result
    assert Spec.SeriesID is not None
    assert len(Spec.SeriesID) > 0
    assert len(Spec.BlockNames) > 0
    assert Spec.Blocks.shape[0] == len(Spec.SeriesID)


def test_data_shape(pipeline_result):
    """Transformed data matrix has the right dimensionality."""
    _, Spec, X, Time, Z, _ = pipeline_result
    T, N = X.shape
    assert N == len(Spec.SeriesID), "Columns must equal number of series in spec"
    assert T == len(Time), "Row count must match Time vector length"
    assert T > 0


def test_result_structure(pipeline_result):
    """ResultObject exposes the expected Kalman-smoother keys."""
    _, Spec, X, _, _, res_object = pipeline_result
    result = res_object.result
    for key in ("x_sm", "X_sm", "Z", "C", "R", "A", "Q", "loglik"):
        assert key in result, f"Missing key '{key}' in DFM result"


def test_loglik_finite(pipeline_result):
    """Log-likelihood values must all be finite (no NaN / Inf blow-up).

    The EM loop seeds LL with [-inf] as a sentinel, so we skip index 0.
    """
    _, _, _, _, _, res_object = pipeline_result
    loglik_history = res_object.result["loglik"]
    assert len(loglik_history) > 1
    actual_values = loglik_history[1:]  # skip the -inf sentinel at index 0
    assert all(
        np.isfinite(v) for v in actual_values
    ), "Non-finite log-likelihood detected — numerical instability in EM."


def test_smoothed_state_shape(pipeline_result):
    """Smoothed state matrix dimensions must be consistent with the data."""
    _, Spec, X, _, _, res_object = pipeline_result
    T = X.shape[0]
    x_sm = res_object.result["x_sm"]
    assert x_sm.shape[0] == T, "Smoothed state row count must match data rows"
    assert x_sm.shape[1] == len(
        Spec.SeriesID
    ), "Smoothed state column count must match number of series"


def test_no_allnan_columns(pipeline_result):
    """After the EM run, no column of the smoothed output should be all-NaN."""
    _, _, _, _, _, res_object = pipeline_result
    x_sm = res_object.result["x_sm"]
    all_nan_cols = np.all(np.isnan(x_sm), axis=0)
    assert not np.any(
        all_nan_cols
    ), f"Columns {np.where(all_nan_cols)[0].tolist()} are all-NaN in smoothed output."
