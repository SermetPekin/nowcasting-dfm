"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional, Union
import numpy as np

#
from dfm_sp.sp_utils import get_latest


# ===================================== FrozenOptions =================================
@dataclass(frozen=True)
class FrozenOptions:
    root: Union[Path, str]
    max_iter: int
    threshold: float
    use_numba: bool
    spec_file_name: str
    country: str
    sample_start: str
    vintage: Optional[str]
    data_folder: Optional[str]

    def __hash__(self):
        root_str = str(self.root) if isinstance(self.root, Path) else self.root
        data_folder_str = (
            str(self.data_folder)
            if isinstance(self.data_folder, Path)
            else self.data_folder
        )
        hashable_attrs = (
            root_str,
            self.max_iter,
            round(self.threshold, 5),
            self.spec_file_name,
            self.sample_start,
            self.vintage,
            data_folder_str,
        )
        return hash(hashable_attrs)


# ===================================== Options =================================
@dataclass(frozen=False, eq=False)
class Options:
    root: Union[Path, str] = Path(".")
    max_iter: int = 5000
    threshold: float = 1e-3
    spec_file_name: str = "Spec_US_example.xls"
    country: str = "US"
    sample_start: str = "2000-01-01"
    vintage: Optional[str] = "auto"
    data_folder: Optional[str] = None
    data_file_name_format: Optional[callable] = (
        None  # e.g. lambda options : f"{options.vintage}-snap.xls"
    )
    out_folder: Union[Path, str] = Path(".")
    use_cache: bool = True
    use_numba: bool = False
    verbose: bool = True
    frozen: Optional[FrozenOptions] = None
    plot1_series: Tuple[str] = tuple(
        ["INDPRO", "HOUST", "PAYEMS", "CPIAUCSL", "UNRATE"]
    )
    plot2_series: Tuple[Tuple[str]] = tuple(
        [("INDPRO", "HOUST"), ("PAYEMS", "CPIAUCSL"), ("UNRATE", "INDPRO")]
    )

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        if self.data_folder is None:
            self.data_folder = self.root / "data" / self.country

        if isinstance(self.spec_file_name, str):
            self.spec_file_name = self.root / self.spec_file_name
        if isinstance(self.sample_start, str):
            self.sample_start = pd.Timestamp(self.sample_start)

        if isinstance(self.out_folder, str):
            self.out_folder = Path(self.out_folder)

        self.frozen = FrozenOptions(
            root=self.root,
            max_iter=self.max_iter,
            threshold=self.threshold,
            use_numba=self.use_numba,
            spec_file_name=self.spec_file_name,
            country=self.country,
            sample_start=self.sample_start,
            vintage=self.vintage,
            data_folder=self.data_folder,
        )

    def get_verbose(self, verbose: Optional[bool] = None) -> bool:
        verbose = self.verbose if verbose is None else verbose
        return verbose

    def name_format(self):
        if self.max_iter < 5000:
            ek = f"TEST-RUN-with-max_iter{self.max_iter}"
        else:
            ek = f"M-{self.max_iter}"
        return (
            f"{self.country}_V-{self.vintage_date}_{ek}_T{self.threshold}-{self.hash()}"
        )

    def hash(self):
        import hashlib
        import pickle

        obj_bytes = pickle.dumps(self.frozen)
        return hashlib.sha256(obj_bytes).hexdigest()[:7]

    @property
    def vintage_date(self):
        vintage = self.vintage
        if vintage is None:
            return get_latest(self.data_folder)

        vintage_lower = str(vintage).lower()
        if vintage_lower in ["auto", "latest"]:
            return get_latest(self.data_folder)

        if isinstance(vintage, int):
            return get_latest(self.data_folder, index=vintage)

        return vintage

    def check(self):
        if self.max_iter < 5000:
            print(
                f"Program will be running with max_iter less than 5000! {self.max_iter}"
            )
            import time

            time.sleep(2)

    def __str__(self):
        return (
            f"DFM Runtime Options\n"
            f"----------------------------------------\n"
            f"Country         : {self.country}\n"
            f"Root Path       : {self.root}\n"
            f"Spec File       : {Path(self.spec_file_name).name}\n"
            f"Data Folder     : {self.data_folder}\n"
            f"Vintage         : {self.vintage_date} (Requested: {self.vintage})\n"
            f"Sample Start    : {self.sample_start.date() if isinstance(self.sample_start, pd.Timestamp) else self.sample_start}\n"
            f"Max Iterations  : {self.max_iter}\n"
            f"EM Threshold    : {self.threshold}\n"
            f"Use Cache       : {self.use_cache}\n"
            f"Hash Key        : {self.hash()}\n"
            f"----------------------------------------"
        )

    def __repr__(self):
        return f"<Options(country='{self.country}', max_iter={self.max_iter}, vintage='{self.vintage_date}', cache={self.use_cache})>"

    def copy(self):
        """Create a copy of the Options object."""
        import copy

        return copy.copy(self)


@dataclass
class ResultObject:
    result: Any
    spec: Any
    options: Options

    def write(self, filename=None):
        if filename is None:
            filename = f"Results-{self.options.country}-Vintage[{self.options.vintage_date}].xlsx"
        with pd.ExcelWriter(filename) as writer:
            self.info().to_excel(writer, sheet_name="Info", index=False)
            for k, v in self.result.items():
                if isinstance(v, np.ndarray):
                    d = pd.DataFrame(v)
                    d.to_excel(writer, sheet_name=str(k))

    def info(self):
        template = """
    #   Res - structure of model results with the following fields
    #       . X_sm | Kalman-smoothed data where missing values are replaced by their expectation
    #       . Z | Smoothed states. Rows give time, and columns are organized according to Res.C.
    #       . C | Observation matrix. The rows correspond
    #          to each series, and the columns are organized as shown below:
    #         - 1-20: These columns give the factor loa dings. For example, 1-5
    #              give loadings for the first block and are organized in
    #              reverse-chronological order (f^G_t, f^G_t-1, f^G_t-2, f^G_t-3,
    #              f^G_t-4). Columns 6-10, 11-15, and 16-20 give loadings for
    #              the second, third, and fourth blocks respectively.
    #       .R: Covariance for observation matrix residuals
    #       .A: Transition matrix. This is a square matrix that follows the
    #      same organization scheme as Res.C's columns. Identity matrices are
    #      used to account for matching terms on the left and righthand side.
    #      For example, we place an I4 matrix to account for matching
    #      (f_t-1; f_t-2; f_t-3; f_t-4) terms.
    #       .Q: Covariance for transition equation residuals.
    #       .Mx: Series mean
    #       .Wx: Series standard deviation
    #       .Z_0: Initial value of state
    #       .V_0: Initial value of covariance matrix
    #       .r: Number of common factors for each block
    #       .p: Number of lags in transition equation
        """
        lines = [line.strip() for line in template.split("\n") if line.strip()]
        return pd.DataFrame(lines, columns=["Info"])
