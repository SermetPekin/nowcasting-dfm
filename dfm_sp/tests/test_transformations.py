"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import numpy as np
from dfm_sp.core.load_data import transformData
import warnings


class MockSpec:
    def __init__(self, transformations, frequencies, series_names=None):
        self.Transformation = transformations
        self.Frequency = frequencies
        n = len(transformations)
        self.SeriesName = series_names or [f"Series_{i}" for i in range(n)]
        self.SeriesID = self.SeriesName


def test_transformations_basic():
    """Test standard application of transformation formulas for monthly data."""
    # 24 months of data
    T = 24
    N = 3
    Z = np.ones((T, N))
    Z[:, 0] = np.arange(1, T + 1)  # lin (1, 2, 3, 4...)
    Z[:, 1] = np.arange(1, T + 1) * 10  # chg (10, 20, 30...)
    Z[:, 2] = np.exp(np.arange(1, T + 1) * 0.1)  # log

    Time = np.arange(T)  # Mock time array

    spec = MockSpec(transformations=["lin", "chg", "log"], frequencies=["m", "m", "m"])

    # transformData drops the first 3 observations (quarter 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore stationarity warnings
        X, Time_out, Z_out = transformData(Z, Time, spec)

    assert X.shape == (T - 3, N)

    # Series 0: "lin" -> Should exactly match Z (from index 3 onward)
    np.testing.assert_allclose(X[:, 0], Z[3:, 0])

    # Series 1: "chg" -> Differences should be constant 10
    # The first difference at index 3 in X matches Z[4] - Z[3] ? No, X[i] is Z[i]-Z[i-1] for monthly.
    np.testing.assert_allclose(X[:, 1], np.full(T - 3, 10.0))

    # Series 2: "log" -> Should be linear 0.1 step
    expected_log = np.arange(4, T + 1) * 0.1
    np.testing.assert_allclose(X[:, 2], expected_log, rtol=1e-5)


def test_transformations_edge_cases():
    """Test edge cases: Zeros, negative values, and existing NaNs."""
    T = 20
    N = 4
    Z = np.ones((T, N)) * 100

    # Inject edge cases
    # 1. Zeros in "pch" (percent change)
    Z[5, 0] = 0.0  # T=5 is index 5. Div by zero when computing pch for index 6.

    # 2. Negative values in "log"
    Z[6, 1] = -50.0

    # 3. Existing NaNs in "ch1" (year over year)
    Z[2, 2] = np.nan
    Z[15, 2] = np.nan

    # 4. Perfectly flat series leading to zero variance (stationarity test edge case)
    # Z[:, 3] is already all 100s

    Time = np.arange(T)
    spec = MockSpec(
        transformations=["pch", "log", "ch1", "lin"], frequencies=["m", "m", "m", "m"]
    )

    with warnings.catch_warnings():
        # We expect some runtime warnings from numpy (divide by zero, invalid log)
        warnings.simplefilter("ignore")
        X, Time_out, Z_out = transformData(Z, Time, spec)

    # 'pch' div by zero at Z[5]=0 -> X[6] (which is index 3 in X, since we drop 3) is predicting Z[6]/Z[5] - 1
    # 100 / 0.0 - 1 = inf
    assert np.isinf(X[3, 0]) or np.isnan(X[3, 0])

    # 'log' of negative number -> NaN
    assert np.isnan(X[3, 1])  # X[3] corresponds to Z[6] because of the 3-period drop

    # 'ch1' YoY change. Z[2] is NaN but that gets dropped in the output anyway (because ch1 requires 12 months prior).
    # X[12, 2] corresponds to Z[15], which we set to NaN.
    assert np.isnan(X[12, 2])

    # Flat series is just flat
    np.testing.assert_allclose(X[:, 3], 100.0)


def test_quarterly_frequency_alignment():
    """Test transformation alignment for 'q' frequency."""
    T = 24
    N = 2
    Z = np.ones((T, N))

    # Step for quarterly is 3
    # Z[2::3] will be the quarterly observation
    Z[2::3, 0] = np.array([10, 20, 30, 40, 50, 60, 70, 80])

    Time = np.arange(T)
    spec = MockSpec(transformations=["chg", "lin"], frequencies=["q", "m"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, Time_out, Z_out = transformData(Z, Time, spec)

    # Quarterly diffs should show values every 3 months.
    # The output X drops the first 3 indices.
    # We expect `X[t::3, 0]` to be the quarterly diffs, others should be NaN.
    # Nans should be well propagated for non-quarter-end months.

    # Let's verify NaNs for non-quarter months
    # Original Z indices available for 'q': 2, 5, 8, 11...
    # Transformed quarterly diff puts first diff at index 5.
    # X drops index 0,1,2. So X[2] is original 5.
    # X[2] = 20 - 10 = 10.
    assert X[2, 0] == 10.0

    # Intervening months should be NaN
    assert np.isnan(X[0, 0])
    assert np.isnan(X[1, 0])
