"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

from pathlib import Path
import os
from typing import List, Tuple
import time
from datetime import date, datetime
from typing import Optional


class Timer:
    def __init__(self):
        pass

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(
                f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run."
            )
            return result

        return wrapper


def get_attr_from_spec(spec, series_id: str, attr_name="SeriesName"):
    for i, s in enumerate(spec.SeriesID):
        if s == series_id:
            items = getattr(spec, attr_name)
            return items[i]
    raise ValueError("Not found attr from spec ")


def get_ok_files(_folder: Path):
    def ok(f: str):
        return f.endswith(".xlsx") or f.endswith(".xls")

    files = os.listdir(_folder)
    ok_files = [x for x in files if ok(x)]
    return ok_files


def as_date(x: str) -> Optional[datetime]:
    try:
        a = datetime.strptime(x, "%Y-%m-%d")
        return a
    except Exception:
        print("ignoring this file", x)
        return None


def get_ok(_folder) -> List[Tuple[date, str]]:
    o = get_ok_files(_folder)
    names_tuple = [(x, x.split(".")[0], x.split(".")[1]) for x in o]
    dates = tuple(as_date(x[1]) for x in names_tuple)
    ok_files = []
    for a, b in zip(dates, names_tuple):
        if a:
            ok_files.append((a, b[0]))
    return ok_files


def get_latest(_folder, index=0) -> str:
    o = get_ok(_folder)
    sorted_d = sorted(o, key=lambda x: x[0], reverse=True)
    if index >= len(sorted_d):
        raise IndexError(
            f"Index {index} is out of range. The list has only {len(sorted_d)} elements. check vintage index. 0 for the latest file. 1 for the one before the latest etc.!"
        )
    return sorted_d[index][1].split(".")[0]
