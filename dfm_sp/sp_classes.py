from dataclasses import dataclass
from typing import Any
from pathlib import Path
import pandas as pd
from dfm_sp.sp_utils import get_latest
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Union


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
    out_folder: Union[Path, str] = Path(".")
    use_cache: bool = True  # Added so user can force re-run
    use_numba: bool = False
    frozen: FrozenOptions = None
    plot1_series: Tuple[str] = tuple(
        ["INDPRO", "HOUST", "PAYEMS", "CPIAUCSL", "UNRATE"]
    )
    plot2_series: Tuple[Tuple[str]] = tuple(
        [("INDPRO", "HOUST"), ("PAYEMS", "CPIAUCSL"), ("UNRATE", "INDPRO")]
    )

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
        vintage_date = self.vintage
        if self.vintage in [None, "auto", "AUTO", "LATEST", "latest"]:
            vintage_date = get_latest(self.data_folder)
        return vintage_date

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        if self.data_folder is None:
            self.data_folder = self.root / "data" / self.country
        if isinstance(self.spec_file_name, str):
            self.spec_file_name = self.root / self.spec_file_name
        if isinstance(self.sample_start, str):
            self.sample_start = pd.Timestamp(self.sample_start)

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

    def check(self):
        if self.max_iter < 5000:
            print(
                f"Program will be running with max_iter less than 5000! {self.max_iter}"
            )
            import time

            time.sleep(2)

    def __str__(self):
        """Create a beautiful string representation for notebooks and terminal output."""
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
        """Short representation showing the core parameters"""
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
