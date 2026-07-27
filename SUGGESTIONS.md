# Improvement Suggestions — `nowcasting-dfm`

Observations and actionable suggestions based on a review of the codebase as of 2026-07-27.
Items are grouped by category and roughly ordered by priority.

---

## 1. Bugs / Correctness

### 1.1 Debug `print` left in `LoadSpec._init_from_file`
[dfm_sp/core/load_spec.py](dfm_sp/core/load_spec.py) line 225 emits a verbose line for every
column it reads (`print(f"setting field: {field} as {raw[field]}")`).
This is noise for every user who loads a spec file. Remove it or gate it behind a `verbose` flag.

### 1.2 Duplicate imports in `__init__.py`
`dfm` and `summarize` are imported twice in [dfm_sp/__init__.py](dfm_sp/__init__.py)
(lines 6–7 and again 19–20). The first pair is dead code. Remove the duplicates.

### 1.3 Internal symbols leak into the public namespace
`daily_report`, `get_latest`, `Timer`, `sp_update_nowcast`, and `Path` are imported at
module level in `__init__.py` but are absent from `__all__`. They are invisibly accessible
to any user who does `from dfm_sp import *` or relies on IDE auto-complete.
Either add them to `__all__` (if intentionally public) or move them to lazy imports inside
the functions that need them.

### 1.4 `Options.frozen` typed incorrectly
```python
# current
frozen: FrozenOptions = None

# should be
frozen: Optional[FrozenOptions] = None
```
Pylance/mypy flag this as a type error.

### 1.5 `run()` and `run_with_options()` missing type annotations
```python
# current
def run(X, Spec, run_options: Options = None) -> ResultObject:
def run_with_options(options: Options, verbose=True):
```
`X` and `Spec` are untyped; `run_with_options` has no return type. Add `np.ndarray`,
`LoadSpec`, and the tuple return type so the linter can follow calls through.

---

## 2. Code Quality

### 2.1 Property naming convention on `LoadSpec`
Python convention is `snake_case` for instance attributes and properties.
`SeriesID`, `SeriesName`, `BlockNames` etc. are PascalCase, which reads like class names.
Consider renaming to `series_id`, `series_name`, `block_names` etc.
The rename can be done safely with the language-server rename tool (as was done for
`load_spec → LoadSpec`). Keep backward-compat aliases if you have downstream users.

### 2.2 Suspicious import in `sp_cache.py`
```python
from numpy.testing import verbose  # line 16
```
`verbose` from `numpy.testing` is almost certainly unused. Verify and remove.

### 2.3 Plotting module fragmentation
`sp_plots.py`, `sp_plots2.py`, `sp_plots3.py` suggest the module grew incrementally.
Consider consolidating into a single `sp_plots.py` with clearly named sections, or splitting
by concern (e.g. `sp_plots_factors.py`, `sp_plots_data.py`) rather than by iteration number.

### 2.4 Non-Pythonic filename
`dfm_sp/core/update_Nowcast.py` uses mixed case in a filename. Python modules are expected
to be all-lowercase. Rename to `update_nowcast.py`.

### 2.5 `benchmark_numba.py` in the project root
Root-level scripts that are not entry points should live in `benchmarks/` or `examples/`.

---

## 3. Testing

### 3.1 No coverage measurement
`pytest-cov` is not in the dev dependencies. Add it and enforce a minimum threshold in CI:
```toml
# pyproject.toml
[tool.coverage.run]
source = ["dfm_sp"]

[tool.pytest.ini_options]
addopts = "--cov=dfm_sp --cov-fail-under=70"
```
```toml
# dependency-groups
dev = [
    ...
    "pytest-cov>=5.0.0",
]
```

### 3.2 No end-to-end / integration test
The test suite (`test_dfm_core.py`, `test_kalman_filter.py`, etc.) tests components in
isolation. A single integration test that runs `run_with_options` → `run` on the sample
data and asserts the nowcast is within a plausible range would catch regressions that unit
tests miss.

### 3.3 `tox.ini` envlist vs CI matrix are out of sync
`tox.ini` lists `py310, py314` but CI tests `3.10–3.13`. Tox is no longer used in CI
(fixed in a previous session), but if tox is kept for local use, align the envlist:
```ini
envlist = py310, py311, py312, py313, py314
```

---

## 4. CI / CD

### 4.1 No linting or type-checking step
`ruff` is already a dev dependency. Add a job (or steps) to the workflow:
```yaml
- name: Lint
  run: uv run ruff check dfm_sp/

- name: Type check
  run: uv run mypy dfm_sp/ --ignore-missing-imports
```
`mypy` (or `pyright`) should be added to the dev dependencies.

### 4.2 No PyPI publish workflow
Add a separate `publish.yml` that triggers on a version tag (`v*.*.*`), builds the wheel,
and uploads to PyPI via `uv publish` (using a `PYPI_TOKEN` secret). Keeps releases
intentional and automated.

### 4.3 Python 3.14 in classifiers but not in CI matrix
Either add `"3.14"` to the matrix or remove it from the classifiers until it is tested.

---

## 5. Developer Experience

### 5.1 No `py.typed` marker (PEP 561)
Add an empty `dfm_sp/py.typed` file and include it in the wheel. This tells downstream
type-checkers that the package ships inline types:
```toml
# pyproject.toml  [tool.hatch.build.targets.wheel]
artifacts = [
    "dfm_sp/py.typed",
    ...
]
```

### 5.2 No pre-commit hooks
A `.pre-commit-config.yaml` running `ruff` and `ruff format` on commit prevents style
issues from reaching CI. Keeps the diff history clean.

### 5.3 `.func_caches/` not in `.gitignore`
The `CacheHandler` writes to `.func_caches/` by default. Verify this directory is
ignored so cached pickle files are never accidentally committed.

### 5.4 No `CHANGELOG.md`
Users upgrading between versions have no record of what changed. Start keeping one,
even informally, from `v0.1.3` onward. Tools like `git-cliff` can auto-generate it from
conventional commits.

---

## 6. API / Features

### 6.1 No progress bar for long EM runs
`max_iter=5000` can take minutes with no feedback. A simple `tqdm` progress bar inside
the EM loop (gated by a `verbose` flag already present in some functions) would
significantly improve the user experience.

### 6.2 `LoadSpec.validate()` method
Add an explicit validation method that can be called after construction to surface all
problems at once rather than letting them propagate into the Kalman filter:
- All series load on the global block (column 0 = 1) — already checked
- No duplicate `SeriesID` values
- `Frequency` values are all within the known set (`d`, `w`, `m`, `q`, `sa`, `a`)
- `Transformation` codes are all registered in `MacroTransformations`

### 6.3 `ResultObject` is not serializable / saveable
Users who run a long EM estimation have no way to save the result to disk and reload it
later without re-running. Consider adding `result.save(path)` / `ResultObject.load(path)`
helpers backed by `pickle` or `numpy.savez`.

### 6.4 Direct FRED API integration
Currently users must either download the sample data bundle or supply their own CSV/XLS.
A thin optional wrapper around the `fredapi` library (already common in econometrics
workflows) would let users pull fresh series directly:
```python
# aspirational API
options = Options(..., fred_api_key="...")
Spec, X, Time, Z = run_with_options(options)   # fetches live from FRED
```

---

## Summary Table

| # | Area | Effort | Impact |
|---|------|--------|--------|
| 1.1 | Remove debug `print` in `LoadSpec` | XS | Medium |
| 1.2 | Deduplicate `__init__.py` imports | XS | Low |
| 1.3 | Fix public namespace leakage | S | Medium |
| 1.4 | Fix `Optional[FrozenOptions]` | XS | Low |
| 1.5 | Add type annotations to `run()` | S | Medium |
| 2.1 | `snake_case` properties on `LoadSpec` | M | High |
| 2.2 | Remove unused `numpy.testing` import | XS | Low |
| 2.3 | Consolidate plotting modules | M | Medium |
| 3.1 | Add `pytest-cov` + threshold | S | High |
| 3.2 | Add integration test | M | High |
| 4.1 | Add ruff + mypy to CI | S | High |
| 4.2 | Add PyPI publish workflow | S | High |
| 5.1 | Add `py.typed` marker | XS | Medium |
| 5.2 | Add pre-commit hooks | S | Medium |
| 5.3 | `.func_caches/` in `.gitignore` | XS | Low |
| 6.1 | `tqdm` progress bar for EM | S | High |
| 6.2 | `LoadSpec.validate()` method | M | Medium |
| 6.3 | `ResultObject` save/load | M | Medium |
