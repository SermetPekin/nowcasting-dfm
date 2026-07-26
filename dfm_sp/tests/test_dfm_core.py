import numpy as np
import pytest
from dfm_sp import dfm


class MockDFMSpec:
    """Mock specification object mimicking the structure expected by the DFM function."""

    def __init__(self, N, num_blocks=1):
        self.Blocks = np.ones((N, num_blocks))
        # Keep things simple with monthly data (no quarterly constraints needed for this mathematical test)
        self.Frequency = np.array(["m"] * N)
        self.SeriesName = np.array([f"Series_{i}" for i in range(N)])
        self.SeriesID = np.array([f"S_{i}" for i in range(N)])
        self.BlockNames = np.array([f"Block_{i}" for i in range(num_blocks)])


def test_em_monotonicity():
    """
    Mathematical econometric test: The Expectation-Maximization (EM) algorithm's
    Log-Likelihood must strictly increase (or stay constant) at each iteration.
    """
    N = 5
    T = 40
    np.random.seed(42)

    # Create some structured data with a single common factor
    factor = np.sin(np.linspace(0, 10, T))
    X = np.random.randn(T, N) * 0.1 + factor[:, None]

    Spec = MockDFMSpec(N, num_blocks=1)

    # Run DFM for 10 iterations max
    Res = dfm(X, Spec, threshold=1e-5, max_iter=10)

    loglik_history = Res["loglik"]

    # Ensure it actually ran multiple iterations
    assert len(loglik_history) > 1
    print("\nLog-likelihood history:", loglik_history)

    # Check monotonicity: loglik[i] >= loglik[i-1]
    for i in range(1, len(loglik_history)):
        # We allow a tiny numerical tolerance (1e-6) for machine epsilon floating point noise
        assert (
            loglik_history[i] >= loglik_history[i - 1] - 1e-6
        ), f"Log-Likelihood decreased from {loglik_history[i-1]} to {loglik_history[i]}"


def test_e2e_ragged_edge_imputation():
    """
    Software engineering/Pipeline test: The core purpose of the Nowcasting DFM is to
    handle "ragged edges" (missing data at the tail ends) and impute them completely.
    """
    N = 4
    T = 50
    np.random.seed(42)

    factor = np.sin(np.linspace(0, 10, T))
    X_clean = np.random.randn(T, N) * 0.1 + factor[:, None]
    X_ragged = X_clean.copy()

    # Introduce ragged edges (missing data at the tail due to publication lags)
    # Series 0 is missing the last 2 months
    X_ragged[-2:, 0] = np.nan
    # Series 1 is missing the last 1 month
    X_ragged[-1:, 1] = np.nan
    # Series 2 is missing a month randomly in the middle
    X_ragged[25, 2] = np.nan
    # Series 3 is completely up to date

    Spec = MockDFMSpec(N, num_blocks=1)

    # Run DFM. Using a slightly higher max_iter to give it room to smooth out the NaNs
    Res = dfm(X_ragged, Spec, threshold=1e-5, max_iter=5)

    x_sm = Res["x_sm"]

    # Assert that the output smoothed data has NO NaNs anywhere
    assert not np.isnan(
        x_sm
    ).any(), "The DFM failed to impute all missing values. NaNs found in output."

    # Ensure output dimensions strictly match input dimensions
    assert x_sm.shape == X_ragged.shape, "Output dimensions changed during smoothing"
