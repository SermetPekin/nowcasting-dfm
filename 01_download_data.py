"""
Step 1 — Download sample data
Run this once before any other example.
Downloads US macro vintage data into the data/ directory.
"""

from dfm_sp import download_sample_data

dest = "data"
country = "US"
force = False
proxy = None

download_sample_data(dest=dest, country=country, force=force, proxy=proxy)
