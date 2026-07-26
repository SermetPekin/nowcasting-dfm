"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

from dfm_sp import download_sample_data

dest = "data"
country = "US"
force = False
proxy = None

download_sample_data(dest=dest, country=country, force=force, proxy=proxy)
