"""
Shared pytest fixtures for dfm_sp tests.
The pipeline_result fixture is session-scoped so the full EM run
is executed only once regardless of how many test modules use it.
"""

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SPEC_FILE = PROJECT_ROOT / "Spec_US_example.xls"
VINTAGE = "2016-06-29"
DATA_FILE = PROJECT_ROOT / "data" / "US" / f"{VINTAGE}.xls"


@pytest.fixture(scope="session")
def pipeline_result():
    """Full get_with_options → run pipeline, shared across all test modules."""
    if not SPEC_FILE.exists() or not DATA_FILE.exists():
        pytest.skip("Sample data not present — run download_sample_data() first.")

    from dfm_sp import Options
    from dfm_sp.sp_run import get_with_options, run

    options = Options(
        root=PROJECT_ROOT,
        vintage=VINTAGE,
        country="US",
        spec_file_name=str(SPEC_FILE),
        max_iter=10,
        threshold=1e-3,
        use_cache=False,
        use_numba=False,
    )

    Spec, X, Time, Z = get_with_options(options, verbose=False)
    res_object = run(X, Spec, options)
    return options, Spec, X, Time, Z, res_object
