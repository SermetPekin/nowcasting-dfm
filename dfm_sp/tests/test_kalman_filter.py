"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""


import numpy as np
import pytest
from dfm_sp.core.dfm import run_kalman_filter_loop, run_kalman_filter_loop_python


def test_kalman_numba_vs_python():
    """
    Test that the Numba JIT compiled Kalman filter loop produces identical results
    to the pure Python implementation, including edge cases with missing data (NaNs).
    """
    nobs = 50
    m = 3  # number of factors/states
    k = 4  # number of observed variables

    np.random.seed(42)
    # Create synthetic observation data Y (k by nobs)
    Y = np.random.randn(k, nobs)

    # Introduce some NaNs to simulate ragged edges / missing data in macro series
    Y[0, 10:15] = np.nan
    Y[2, 45:] = np.nan
    Y[:, 25] = np.nan  # an entirely missing column

    # Transition matrix A
    A = np.eye(m) * 0.9 + np.random.randn(m, m) * 0.05
    # Observation matrix C
    C = np.random.randn(k, m)
    # Transition error covariance Q
    Q = np.eye(m) * 0.1
    # Observation error covariance R
    R = np.eye(k) * 0.2

    # Initial state
    Zu = np.zeros(m)
    Vu = np.eye(m)

    # 1. Run Python version
    Zm_p, Vm_p, ZmU_p, VmU_p, loglik_p = run_kalman_filter_loop_python(
        nobs, m, Y, A, C, Q, R, Zu.copy(), Vu.copy()
    )

    # 2. Run Numba version
    Zm_n, Vm_n, ZmU_n, VmU_n, loglik_n = run_kalman_filter_loop(
        nobs, m, Y, A, C, Q, R, Zu.copy(), Vu.copy()
    )

    # 3. Assert outputs are exactly the same up to 1e-8 precision
    np.testing.assert_allclose(Zm_n, Zm_p, rtol=1e-7, atol=1e-8, err_msg="Zm mismatch")
    np.testing.assert_allclose(Vm_n, Vm_p, rtol=1e-7, atol=1e-8, err_msg="Vm mismatch")
    np.testing.assert_allclose(
        ZmU_n, ZmU_p, rtol=1e-7, atol=1e-8, err_msg="ZmU mismatch"
    )
    np.testing.assert_allclose(
        VmU_n, VmU_p, rtol=1e-7, atol=1e-8, err_msg="VmU mismatch"
    )
    assert np.isclose(loglik_n, loglik_p, rtol=1e-7, atol=1e-8), "loglik mismatch"
