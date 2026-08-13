"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

FRED-MD / FRED-QD integration for nowcasting-dfm
=================================================
Downloads and parses monthly FRED-MD or quarterly FRED-QD vintages from the
St. Louis Fed public archive and returns them in the same (Spec, X, Time, Z)
format as get_with_options(), so they drop straight into run().

Reference
---------
McCracken, M. W. & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic
Research. Journal of Business & Economic Statistics, 34(4), 574–589.
https://doi.org/10.1080/07350015.2015.1086655

Dataset archive
---------------
https://www.stlouisfed.org/research/economists/mccracken/fred-databases
"""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from io import StringIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from dfm_sp.core.load_spec import LoadSpec, SpecConfig

# ── URL templates ────────────────────────────────────────────────────────────

_FREDMD_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
    "research/fred-md/monthly/{vintage}-md.csv"
)
_FREDQD_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
    "research/fred-md/quarterly/{vintage}-qd.csv"
)
_CURRENT_MD = _FREDMD_URL.replace("{vintage}-md", "current")
_CURRENT_QD = _FREDQD_URL.replace("{vintage}-qd", "current")

# ── McCracken & Ng transformation code map ───────────────────────────────────
# tcode → human-readable label (for metadata/reporting)
FREDMD_TCODE_LABELS = {
    1: "Levels",
    2: "First Difference",
    3: "Second Difference",
    4: "Natural Log",
    5: "First Log Difference ×100",
    6: "Second Log Difference ×100",
    7: "Percent Change",
}


# ── Internal helpers ─────────────────────────────────────────────────────────


def _fetch_csv(url: str, proxy: Optional[str] = None) -> str:
    """Download a URL and return its text content.

    Uses curl when available (bypasses Python TLS fingerprinting blocks on
    stlouisfed.org). Falls back to urllib for environments without curl.
    """
    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-sL", "--max-time", "30", url]
        if proxy:
            cmd += ["--proxy", proxy]
        result = subprocess.run(cmd, capture_output=True, timeout=35)
        if result.returncode != 0:
            raise RuntimeError(
                f"curl failed (exit {result.returncode}): {result.stderr.decode()}"
            )
        return result.stdout.decode("utf-8")

    # urllib fallback — may be blocked by some servers' TLS fingerprinting
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "nowcasting-dfm"})
    opener = urllib.request.build_opener()
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    with opener.open(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _validate_csv(content: str, url: str) -> None:
    """Raise a clear error if the server returned HTML instead of CSV."""
    first = content.lstrip()[:10].lower()
    if first.startswith("<!") or first.startswith("<html"):
        raise RuntimeError(
            f"The St. Louis Fed returned an HTML page instead of CSV for:\n  {url}\n"
            "This vintage may not be individually available on the server.\n"
            "Only vintages from approximately 2025 onward are accessible via direct URL.\n"
            "For older vintages, download the CSV manually from:\n"
            "  https://www.stlouisfed.org/research/economists/mccracken/fred-databases\n"
            "Then load it with: load_fredmd_file('path/to/file.csv')"
        )


def _resolve_url(vintage: str, quarterly: bool) -> str:
    template = _FREDQD_URL if quarterly else _FREDMD_URL
    current = _CURRENT_QD if quarterly else _CURRENT_MD
    if vintage.lower() == "current":
        return current
    ym = vintage[:7]  # accept "YYYY-MM" or "YYYY-MM-DD"
    return template.format(vintage=ym)


def _parse_csv(content: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return (raw_data, tcodes, series_names).

    raw_data : DataFrame  — dates as index, series as columns
    tcodes   : int array  — one McCracken & Ng tcode per column
    """
    df = pd.read_csv(StringIO(content), header=0, low_memory=False)

    date_col = df.columns[0]  # "sasdate"
    series_cols = [c for c in df.columns[1:] if not c.startswith("Unnamed")]

    # The first data row holds the transformation codes ("Transform:,5,5,…")
    tcode_row = df.iloc[0]
    tcodes = pd.to_numeric(tcode_row[series_cols], errors="coerce").fillna(1).astype(int).values

    data = df.iloc[1:].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data = data.set_index(date_col).sort_index()
    data = data[series_cols].apply(pd.to_numeric, errors="coerce")

    return data, tcodes, series_cols


def _apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """Apply a single McCracken & Ng transformation code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if tcode == 1:
            return series
        elif tcode == 2:
            return series.diff()
        elif tcode == 3:
            return series.diff().diff()
        elif tcode == 4:
            return np.log(series)
        elif tcode == 5:
            return np.log(series).diff() * 100
        elif tcode == 6:
            return np.log(series).diff().diff() * 100
        elif tcode == 7:
            return (series / series.shift(1) - 1) * 100
        else:
            warnings.warn(f"Unknown FRED-MD tcode {tcode}; treating as levels.")
            return series


def _build_spec(series_names: List[str], tcodes: np.ndarray, frequency: str) -> LoadSpec:
    """Construct a LoadSpec with a single global factor block."""
    N = len(series_names)
    config = SpecConfig(
        series_id=series_names,
        series_name=series_names,
        frequency=[frequency] * N,
        units=[FREDMD_TCODE_LABELS.get(int(tc), "") for tc in tcodes],
        # data is pre-transformed; DFM sees it as levels
        transformation=["lin"] * N,
        category=["FRED-MD"] * N,
        block_names=["Global"],
        blocks_matrix=np.ones((N, 1), dtype=int),
    )
    return LoadSpec(config)


# ── Public API ────────────────────────────────────────────────────────────────


_FREDMD_PAGE = "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"


def fredmd_current_vintage(quarterly: bool = False, proxy: Optional[str] = None) -> str:
    """Return the ``"YYYY-MM"`` release label for the current FRED-MD vintage.

    Scraped from the St. Louis Fed database page, where the ``current.csv``
    link explicitly maps to the dated file (e.g. ``2026-07-md.csv``).
    Use the returned string directly in output file names.

    Examples
    --------
    >>> v = fredmd_current_vintage()          # e.g. "2026-07"
    >>> result.write(f"Results-FredDb-{v}")
    """
    freq_tag = "qd" if quarterly else "md"
    pattern = re.compile(rf"(?:monthly|quarterly)/(\d{{4}}-\d{{2}})-{freq_tag}\.csv")

    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-sL", "--max-time", "15", _FREDMD_PAGE]
        if proxy:
            cmd += ["--proxy", proxy]
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        if result.returncode == 0:
            m = pattern.search(result.stdout.decode("utf-8", errors="replace"))
            if m:
                return m.group(1)

    raise RuntimeError(
        "Could not determine the current FRED-MD vintage from the St. Louis Fed page."
    )


def load_fredmd_vintage(
    vintage: str = "current",
    sample_start: Optional[str] = None,
    series: Optional[List[str]] = None,
    quarterly: bool = False,
    proxy: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[LoadSpec, np.ndarray, np.ndarray, np.ndarray]:
    """Download and parse a FRED-MD (or FRED-QD) vintage.

    Returns the same ``(Spec, X, Time, Z)`` tuple as ``get_with_options()``,
    so the result drops directly into ``run()``.

    Parameters
    ----------
    vintage : str
        ``"YYYY-MM"`` for a specific release (e.g. ``"2016-12"``), or
        ``"current"`` for the latest available file.
        Vintage denotes the *release month*, not the last observation period.
        Files from 2015-01 onward are available individually; earlier vintages
        are distributed as a bulk archive on the St. Louis Fed website.
    sample_start : str, optional
        Drop observations before this date, e.g. ``"2000-01-01"``.
    series : list of str, optional
        Subset of FRED-MD series IDs to load (e.g. ``["INDPRO", "UNRATE"]``).
        All ~130 series are included when *series* is ``None``.
    quarterly : bool
        If ``True``, load FRED-QD (quarterly) instead of FRED-MD (monthly).
    proxy : str, optional
        HTTP/HTTPS proxy URL.
    verbose : bool
        Print download and parsing progress (default ``True``).

    Returns
    -------
    Spec : LoadSpec
    X    : np.ndarray of shape (T, N) — transformed (stationary) panel
    Time : np.ndarray of shape (T,)   — datetime64[ns] observation dates
    Z    : np.ndarray of shape (T, N) — raw (untransformed) panel

    Examples
    --------
    >>> from dfm_sp import load_fredmd_vintage, Options, run
    >>> Spec, X, Time, Z = load_fredmd_vintage("2016-12")
    >>> result = run(X, Spec, Options(vintage="2016-12"))
    >>> result.result  # EM-estimated parameters

    Load a subset of series only:

    >>> Spec, X, Time, Z = load_fredmd_vintage(
    ...     "current", series=["INDPRO", "UNRATE", "PAYEMS", "CPIAUCSL"]
    ... )
    """
    url = _resolve_url(vintage, quarterly)
    if verbose:
        print(f"[FRED-MD] Downloading vintage '{vintage}' from St. Louis Fed...")
    content = _fetch_csv(url, proxy)
    _validate_csv(content, url)
    if verbose:
        print(f"[FRED-MD] Parsing CSV...")

    return _apply_fredmd_pipeline(
        content, sample_start, series, quarterly, verbose, source=f"vintage '{vintage}'"
    )


def load_fredmd_file(
    filepath: str,
    sample_start: Optional[str] = None,
    series: Optional[List[str]] = None,
    quarterly: bool = False,
    verbose: bool = True,
) -> Tuple[LoadSpec, np.ndarray, np.ndarray, np.ndarray]:
    """Load a locally saved FRED-MD or FRED-QD CSV file.

    Useful for historical vintages not individually accessible on the server,
    which can be downloaded as a bulk zip from the St. Louis Fed website:
    https://www.stlouisfed.org/research/economists/mccracken/fred-databases

    Parameters and return values are identical to ``load_fredmd_vintage()``.

    Examples
    --------
    >>> Spec, X, Time, Z = load_fredmd_file("2016-12.csv", sample_start="2000-01-01")
    """
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FRED-MD file not found: {filepath}")
    if verbose:
        print(f"[FRED-MD] Reading {path.name}...")
    content = path.read_text(encoding="utf-8")
    _validate_csv(content, str(path))
    if verbose:
        print("[FRED-MD] Parsing CSV...")

    return _apply_fredmd_pipeline(
        content, sample_start, series, quarterly, verbose, source=path.name
    )


def _apply_fredmd_pipeline(
    content: str,
    sample_start: Optional[str],
    series: Optional[List[str]],
    quarterly: bool,
    verbose: bool,
    source: str,
) -> Tuple[LoadSpec, np.ndarray, np.ndarray, np.ndarray]:
    """Shared parsing pipeline used by both load_fredmd_vintage and load_fredmd_file."""
    raw_df, tcodes, all_series = _parse_csv(content)

    if series is not None:
        missing = [s for s in series if s not in raw_df.columns]
        if missing:
            raise ValueError(f"Series not in FRED-MD file '{source}': {missing}")
        idx = [all_series.index(s) for s in series]
        raw_df = raw_df[series]
        tcodes = tcodes[idx]
        selected = list(series)
    else:
        selected = list(all_series)

    X_df = raw_df.copy()
    for col, tc in zip(selected, tcodes):
        X_df[col] = _apply_tcode(raw_df[col], tc)

    max_lags = int(max(
        2 if tc in (3, 6) else 1 if tc in (2, 4, 5, 7) else 0
        for tc in tcodes
    ))
    X_df = X_df.iloc[max_lags:]
    Z_df = raw_df.iloc[max_lags:]

    if sample_start is not None:
        cutoff = pd.Timestamp(sample_start)
        X_df = X_df[X_df.index >= cutoff]
        Z_df = Z_df[Z_df.index >= cutoff]

    valid = X_df.columns[~X_df.isnull().all()].tolist()
    if len(valid) < len(selected):
        dropped = [c for c in selected if c not in valid]
        warnings.warn(
            f"Dropped {len(dropped)} all-NaN series for '{source}': {dropped}",
            stacklevel=3,
        )
        idx_valid = [selected.index(c) for c in valid]
        tcodes = tcodes[idx_valid]
        selected = valid
        X_df = X_df[valid]
        Z_df = Z_df[valid]

    freq = "q" if quarterly else "m"
    Spec = _build_spec(selected, tcodes, freq)
    Time = X_df.index.to_numpy()
    X = X_df.to_numpy(dtype=float)
    Z = Z_df.to_numpy(dtype=float)

    if verbose:
        print(f"[FRED-MD] Ready: {X.shape[1]} series, {X.shape[0]} observations "
              f"({Time[0]} → {Time[-1]})")
    return Spec, X, Time, Z
