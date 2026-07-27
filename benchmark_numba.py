"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

"""
Numba Benchmark
===============
Compares the pure-Python and Numba-JIT Kalman filter implementations:
  1. Verifies both produce numerically identical results.
  2. Reports wall-clock speedup across increasing dataset sizes.

Note: Numba JIT is beneficial only for large datasets (nobs > ~1000).
For typical macroeconomic datasets (~200 monthly observations) the JIT
dispatch overhead dominates and pure Python is faster. The default is
therefore use_numba=False; pass use_numba=True for large panel datasets.

Usage:
    uv run python benchmark_numba.py
"""

import time
import numpy as np
from dfm_sp.core.dfm import run_kalman_filter_loop, run_kalman_filter_loop_python

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(nobs: int, m: int = 5, k: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((k, nobs))
    # Introduce realistic missing data (ragged edges)
    Y[0, : nobs // 10] = np.nan
    Y[-1, -nobs // 10 :] = np.nan

    A = np.eye(m) * 0.9 + rng.standard_normal((m, m)) * 0.02
    C = rng.standard_normal((k, m))
    Q = np.eye(m) * 0.1
    R = np.eye(k) * 0.2
    Zu = np.zeros(m)
    Vu = np.eye(m)
    return nobs, m, Y, A, C, Q, R, Zu, Vu


def _time(fn, *args, runs: int = 5) -> float:
    """Return median wall-clock seconds over `runs` calls."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# 1. Correctness check
# ---------------------------------------------------------------------------


def check_correctness():
    print("=" * 60)
    print("1. CORRECTNESS CHECK")
    print("=" * 60)
    nobs, m, Y, A, C, Q, R, Zu, Vu = _make_inputs(nobs=200)

    res_py = run_kalman_filter_loop_python(nobs, m, Y, A, C, Q, R, Zu.copy(), Vu.copy())
    res_jit = run_kalman_filter_loop(nobs, m, Y, A, C, Q, R, Zu.copy(), Vu.copy())

    labels = ["Zm", "Vm", "ZmU", "VmU", "loglik"]
    all_ok = True
    for label, py_val, jit_val in zip(labels, res_py, res_jit):
        ok = np.allclose(py_val, jit_val, rtol=1e-7, atol=1e-8)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {label:<8} {status}")

    print()
    if all_ok:
        print("  All outputs match within tolerance (rtol=1e-7, atol=1e-8).")
    else:
        print("  WARNING: Outputs differ — investigate before publishing.")
    print()


# ---------------------------------------------------------------------------
# 2. Timing benchmark
# ---------------------------------------------------------------------------


def run_benchmark():
    print("=" * 60)
    print("2. TIMING BENCHMARK  (median of 5 runs each)")
    print("=" * 60)
    print(f"  {'nobs':>6}  {'Python (s)':>12}  {'Numba (s)':>12}  {'Speedup':>10}")
    print("  " + "-" * 46)

    # Warm up the JIT compiler once before timing
    _args = _make_inputs(nobs=50)
    run_kalman_filter_loop(*_args)

    for nobs in [100, 500, 1_000, 5_000, 10_000]:
        args = _make_inputs(nobs=nobs)

        t_py = _time(run_kalman_filter_loop_python, *args)
        t_jit = _time(run_kalman_filter_loop, *args)
        speedup = t_py / t_jit if t_jit > 0 else float("inf")

        print(f"  {nobs:>6,}  {t_py:>12.4f}  {t_jit:>12.4f}  {speedup:>9.1f}x")

    print()


# ---------------------------------------------------------------------------
# 3. Full DFM run comparison (Options-level)
# ---------------------------------------------------------------------------


def run_full_dfm_benchmark():
    print("=" * 60)
    print("3. FULL DFM RUN  (use_numba=True vs False)")
    print("=" * 60)
    try:
        from dfm_sp import Options, get_with_options, run

        options_numba = Options(
            vintage="2016-06-29",
            country="US",
            spec_file_name="Spec_US_example.xls",
            max_iter=50,  # keep short for benchmark purposes
            threshold=1e-3,
            use_cache=False,
            use_numba=True,
        )
        options_plain = options_numba.copy()
        options_plain.use_numba = False

        Spec, X, _, _ = get_with_options(options_numba)

        t0 = time.perf_counter()
        res_numba = run(X, Spec, options_numba)
        t_numba = time.perf_counter() - t0

        t0 = time.perf_counter()
        res_plain = run(X, Spec, options_plain)
        t_plain = time.perf_counter() - t0

        speedup = t_plain / t_numba if t_numba > 0 else float("inf")
        print(f"  use_numba=True  : {t_numba:.3f}s")
        print(f"  use_numba=False : {t_plain:.3f}s")
        print(f"  Speedup         : {speedup:.1f}x")

        # Quick sanity check on loglik
        ll_numba = res_numba.result["loglik"][-1]
        ll_plain = res_plain.result["loglik"][-1]
        ok = np.isclose(ll_numba, ll_plain, rtol=1e-5)
        print(
            f"  loglik match    : {'PASS' if ok else 'FAIL'}  "
            f"(numba={ll_numba:.6f}, plain={ll_plain:.6f})"
        )
    except Exception as exc:
        print(f"  Skipped — data not found or error: {exc}")
    print()


if __name__ == "__main__":
    check_correctness()
    run_benchmark()
    run_full_dfm_benchmark()
