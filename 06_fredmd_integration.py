"""
Step 6 — FRED-MD integration (McCracken & Ng 2016)

Downloads a vintage directly from the St. Louis Fed public archive —
no Excel spec file, no manual data preparation.

The load_fredmd_vintage() function returns the same (Spec, X, Time, Z)
tuple as get_with_options(), so it drops straight into run().

Dataset: https://www.stlouisfed.org/research/economists/mccracken/fred-databases
"""

from dfm_sp import Options, fredmd_current_vintage, load_fredmd_file, load_fredmd_vintage, run, export_fredmd_spec, export_fredmd_data

# ── Option A: download the current (latest) vintage directly ──────────────────
# Only vintages from ~2025 onward are individually accessible via URL.
# Use "current" to always get the most recent release.
CORE_SERIES = None # ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "HOUST", "FEDFUNDS"]

vintage = fredmd_current_vintage()  # e.g. "2025-09"
print(f"Step 1: downloading FRED-MD vintage '{vintage}'...")
Spec, X, Time, Z = load_fredmd_vintage(
    vintage="current",
    sample_start="2000-01-01",
    series=CORE_SERIES,
)

# ── Optional: export an editable Spec Excel file ─────────────────────────────
# The file mirrors Spec_US_example.xls — add block columns, reorder series,
# adjust transformations, then load it with LoadSpec("Spec_FredMD_....xlsx").
spec_path = export_fredmd_spec(vintage, series=CORE_SERIES)
print(f"Editable spec written to: {spec_path}")

# Save the transformed data so it can be reloaded via the standard pipeline.
# Pair with the spec above (Transformation=lin) to avoid double-transforming.
data_path = export_fredmd_data(vintage, series=CORE_SERIES, sample_start="2000-01-01")
print(f"Transformed data written to: {data_path}")

print(f"Step 2: running DFM on {X.shape[1]} series  {X.shape[0]} observations...")
options = Options(max_iter=500, use_cache=False, verbose=True)
result = run(X, Spec, options)
result.write(f"Results-FredDb-{vintage}")


# ── Option B: load a manually downloaded CSV (any vintage) ───────────────────
# Download the CSV from:
#   https://www.stlouisfed.org/research/economists/mccracken/fred-databases
# then pass the local path:
#
# Spec, X, Time, Z = load_fredmd_file(
#     "2016-12.csv",
#     sample_start="2000-01-01",
#     series=CORE_SERIES,
# )
