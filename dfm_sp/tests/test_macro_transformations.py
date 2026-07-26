import numpy as np
import pytest
from dfm_sp.sp_transformations import MacroTransformations


def test_macro_transformations_load_data_logic():
    """
    Simulates the exact array slicing and transformation logic inside load_data.py
    to verify that MacroTransformations lambda math generates the correct matrix shapes
    and values without breaking under the step logic.
    """
    # Simulate a raw input array Z with T=36 observations
    T = 36
    Z_i = np.arange(1, T + 1, dtype=float)  # 1.0, 2.0, ..., 36.0

    # Simulating load_data.py Freq_dict step logic for monthly ("m")
    step = 1
    t1 = step - 1  # 0
    n = step / 12  # 1/12

    # 1. Init empty X array as it does in load_data.py
    X = np.empty(T)
    X[:] = np.nan

    # Get formula library dynamically
    formula_dict = MacroTransformations.get_formulas(t1, step, n)

    # Test 1: "lin" (Levels - applied to whole array)
    transform_lin = formula_dict.get("lin")
    X[:] = transform_lin(Z_i)
    np.testing.assert_array_equal(X, Z_i)

    # Test 2: "chg" (First Difference - requires sliced targeting)
    X[:] = np.nan
    transform_chg = formula_dict.get("chg")
    X[t1::step] = transform_chg(Z_i)

    # "chg" difference of np.arange(1, 37) should be exactly 1.0 for every step > 0
    # X[0] is np.nan because it appends np.nan initially
    assert np.isnan(X[0])
    np.testing.assert_allclose(X[1:], np.ones(T - 1))

    # Test 3: "ch1" (Year over Year - skip 12 periods)
    X[:] = np.nan
    transform_ch1 = formula_dict.get("ch1")
    X[12 + t1 :: step] = transform_ch1(Z_i)

    # "ch1" should be exactly 12.0 for every index >= 12
    assert np.isnan(X[11])
    np.testing.assert_allclose(X[12:], np.full(T - 12, 12.0))

    # Test 4: "dln" (Log Difference - sliced target)
    Z_exp = np.exp(
        np.arange(1, T + 1) * 0.1
    )  # exponential growth so log diff is constant
    X[:] = np.nan
    transform_dln = formula_dict.get("dln")
    X[t1::step] = transform_dln(Z_exp)

    # "dln" multiplies by 100. Diff of 0.1 * 100 = 10.0
    assert np.isnan(X[0])
    np.testing.assert_allclose(X[1:], np.full(T - 1, 10.0), rtol=1e-5)

    # Test 5: "dl1" (YoY Log Difference - skip 12 periods)
    X[:] = np.nan
    transform_dl1 = formula_dict.get("dl1")
    X[12 + t1 :: step] = transform_dl1(Z_exp)

    # "dl1" log difference over 12 periods = 12 * 0.1 * 100 = 120.0
    assert np.isnan(X[11])
    np.testing.assert_allclose(X[12:], np.full(T - 12, 120.0), rtol=1e-5)


def test_macro_transformations_quarterly_logic():
    """
    Test the quarterly data step slicing (step = 3) which modifies the lambda array math.
    """
    T = 24
    Z_i = np.arange(1, T + 1, dtype=float)

    step = 3
    t1 = step - 1  # 2
    n = step / 12  # 3/12 = 0.25

    X = np.empty(T)
    X[:] = np.nan

    formula_dict = MacroTransformations.get_formulas(t1, step, n)

    # "chg" for quarterly: diffs between indices spaced by 3
    # Z_i = 1, 2, 3, 4, 5, 6...
    # Quarterly obs fall on index 2 (val 3.0), 5 (val 6.0), 8 (val 9.0)
    transform_chg = formula_dict.get("chg")
    X[t1::step] = transform_chg(Z_i)

    # First obs at index 2 should be NaN because nothing preceded it
    assert np.isnan(X[2])

    # Second quarterly obs at index 5 should be Z[5] - Z[2] = 6.0 - 3.0 = 3.0
    assert X[5] == 3.0

    # Third quarterly obs at index 8 should be Z[8] - Z[5] = 9.0 - 6.0 = 3.0
    assert X[8] == 3.0
