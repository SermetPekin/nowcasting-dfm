"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import urllib.request
import json
from pathlib import Path

_GITHUB_API = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"

_OWNER = "SermetPekin"
_REPO = "nowcasting-dfm"


def _make_opener(proxy: str | None):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def _list_github_files(remote_path: str, proxy: str | None = None) -> list[dict]:
    url = _GITHUB_API.format(owner=_OWNER, repo=_REPO, path=remote_path)
    req = urllib.request.Request(url, headers={"User-Agent": "dfm_sp"})
    with _make_opener(proxy).open(req) as resp:
        return json.loads(resp.read().decode())


def _download_file(
    download_url: str, local_path: Path, proxy: str | None = None
) -> None:
    """Download a file using the URL provided directly by the GitHub API."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(download_url, headers={"User-Agent": "dfm_sp"})
    with _make_opener(proxy).open(req) as resp, open(local_path, "wb") as f:
        f.write(resp.read())


def download_sample_data(
    dest: str = ".",
    country: str = "US",
    force: bool = False,
    proxy: str | None = None,
) -> Path:
    """Download the sample vintage data and spec file from the GitHub repository.

    Parameters
    ----------
    dest:
        Local directory to download into. Defaults to the current working directory.
        Files are placed at ``<dest>/data/<country>/`` and ``<dest>/Spec_US_example.xls``.
    country:
        Country subfolder to download (default ``"US"``).
    force:
        Re-download even if files already exist (default ``False``).
    proxy:
        Optional proxy URL, e.g. ``"http://user:pass@proxy.example.com:8080"``.
        Applies to both the GitHub API listing and the file downloads.

    Returns
    -------
    Path
        The local ``data/<country>/`` directory.

    Examples
    --------
    >>> from dfm_sp import download_sample_data
    >>> download_sample_data()                          # downloads to ./data/US/
    >>> download_sample_data(dest="/tmp/myproject")
    >>> download_sample_data(proxy="http://proxy:8080") # behind a corporate proxy
    """
    dest_path = Path(dest)
    data_dir = dest_path / "data" / country
    spec_dest = dest_path / "Spec_US_example.xls"

    # --- vintage files ---
    remote_data_path = f"data/{country}"
    print(f"Fetching file list from {_OWNER}/{_REPO}/{remote_data_path} ...")
    entries = _list_github_files(remote_data_path, proxy=proxy)
    files = [e for e in entries if e["type"] == "file"]

    downloaded = 0
    skipped = 0
    for entry in files:
        local_file = data_dir / entry["name"]
        if local_file.exists() and not force:
            skipped += 1
            continue
        print(f"  Downloading {entry['name']} ...")
        _download_file(entry["download_url"], local_file, proxy=proxy)
        downloaded += 1

    # --- spec file ---
    spec_entries = _list_github_files("Spec_US_example.xls", proxy=proxy)
    if not spec_dest.exists() or force:
        print("  Downloading Spec_US_example.xls ...")
        _download_file(spec_entries["download_url"], spec_dest, proxy=proxy)
        downloaded += 1
    else:
        skipped += 1

    print(f"Done. {downloaded} file(s) downloaded, {skipped} already present.")
    return data_dir
