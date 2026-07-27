""" 
BSD 3-Clause License

Copyright (c) 2018, Federal Reserve Bank of New York (original MATLAB implementation by Eric Qian and Brandyn Bok)
Copyright (c) 2019, Galib Khan (independent Python translation, not affiliated with FRBNY, https://github.com/MajesticKhan/Nowcasting-Python)
Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

""" 
from __future__ import annotations  
from typing import Optional, List, Dict, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass

if TYPE_CHECKING:  
    from dfm_sp.sp_transformations import MacroTransformations

@dataclass
class SpecConfig:
    """Alternative way to initialize a Spec object without relying on Excel"""
    series_id: List[str]
    series_name: List[str]
    frequency: List[str]
    units: List[str]
    transformation: List[str]
    category: List[str]
    block_names: List[str]
    blocks_matrix: np.ndarray

class LoadSpec:

    """
    Load model specification for a dynamic factor model (DFM).

    Attributes:
        SeriesID (np.ndarray): Array of series IDs.
        SeriesName (np.ndarray): Array of series names.
        Frequency (np.ndarray): Array of frequencies (e.g., 'd', 'w', 'm').
        Units (np.ndarray): Array of units for each series.
        Transformation (np.ndarray): Array of transformation codes.
        Category (np.ndarray): Array of categories for each series.
        Blocks (np.ndarray): Matrix of block loadings.
        BlockNames (List[str]): Names of the blocks.
        UnitsTransformed (np.ndarray): Human-readable descriptions of transformations.
    """

    def __init__(self, spec_input: Union[str, Path, SpecConfig]):
        self._SeriesID: Optional[np.ndarray] = None
        self._SeriesName: Optional[np.ndarray] = None
        self._Frequency: Optional[np.ndarray] = None
        self._Units: Optional[np.ndarray] = None
        self._Transformation: Optional[np.ndarray] = None
        self._Category: Optional[np.ndarray] = None
        self._Blocks: Optional[np.ndarray] = None
        self._BlockNames: Optional[List[str]] = None
        self._UnitsTransformed: Optional[np.ndarray] = None
        
        if isinstance(spec_input, (str, Path)):
            self._init_from_file(str(spec_input))
        elif isinstance(spec_input, SpecConfig):
            self._init_from_config(spec_input)
        else:
            raise TypeError(
                "spec_input must be a file path (str/Path) or a SpecConfig instance"
            )

    def __str__(self) -> str:
        summary = [
            "Dynamic Factor Model Specification",
            "=" * 40,
            f"Number of series: {len(self.SeriesID)}",
            f"Block names: {', '.join(self.BlockNames)}",
            f"Frequency distribution: {self._get_frequency_distribution()}",
            "\nSeries Details:",
            "-" * 40,
        ]

        # Create a table of series information
        series_df = pd.DataFrame({
            "ID": self.SeriesID,
            "Name": self.SeriesName,
            "Frequency": self.Frequency,
            "Units": self.Units,
            "Transformation": self.Transformation,
            "Category": self.Category,
        })

        # Add block information for each series
        for i, block_name in enumerate(self.BlockNames):
            series_df[f"Block: {block_name}"] = self.Blocks[:, i] if i < self.Blocks.shape[1] else 0

        summary.append(series_df.to_string(index=False))
        return "\n".join(summary)

    def _get_frequency_distribution(self) -> str:
        """Helper method to get frequency distribution as a string."""
        freq_counts = pd.Series(self.Frequency).value_counts().to_dict()
        return ", ".join(f"{freq}: {count}" for freq, count in freq_counts.items())

    def _guard(self, name: str, value: Optional[np.ndarray]) -> np.ndarray:
        if value is None:
            raise RuntimeError(
                f"LoadSpec.{name} is None — the spec was not fully initialised. "
                "Check that your spec file contains the required columns and passes validation."
            )
        return value

    @property
    def SeriesID(self) -> np.ndarray:
        return self._guard("SeriesID", self._SeriesID)

    @SeriesID.setter
    def SeriesID(self, value: np.ndarray) -> None:
        self._SeriesID = value

    @property
    def SeriesName(self) -> np.ndarray:
        return self._guard("SeriesName", self._SeriesName)

    @SeriesName.setter
    def SeriesName(self, value: np.ndarray) -> None:
        self._SeriesName = value

    @property
    def Frequency(self) -> np.ndarray:
        return self._guard("Frequency", self._Frequency)

    @Frequency.setter
    def Frequency(self, value: np.ndarray) -> None:
        self._Frequency = value

    @property
    def Units(self) -> np.ndarray:
        return self._guard("Units", self._Units)

    @Units.setter
    def Units(self, value: np.ndarray) -> None:
        self._Units = value

    @property
    def Transformation(self) -> np.ndarray:
        return self._guard("Transformation", self._Transformation)

    @Transformation.setter
    def Transformation(self, value: np.ndarray) -> None:
        self._Transformation = value

    @property
    def Category(self) -> np.ndarray:
        return self._guard("Category", self._Category)

    @Category.setter
    def Category(self, value: np.ndarray) -> None:
        self._Category = value

    @property
    def Blocks(self) -> np.ndarray:
        return self._guard("Blocks", self._Blocks)

    @Blocks.setter
    def Blocks(self, value: np.ndarray) -> None:
        self._Blocks = value

    @property
    def BlockNames(self) -> List[str]:
        if self._BlockNames is None:
            raise RuntimeError(
                "LoadSpec.BlockNames is None — the spec was not fully initialised."
            )
        return self._BlockNames

    @BlockNames.setter
    def BlockNames(self, value: List[str]) -> None:
        self._BlockNames = value

    @property
    def UnitsTransformed(self) -> np.ndarray:
        return self._guard("UnitsTransformed", self._UnitsTransformed)

    @UnitsTransformed.setter
    def UnitsTransformed(self, value: np.ndarray) -> None:
        self._UnitsTransformed = value

    # --- Initialization Methods ---
    def _init_from_config(self, config: SpecConfig) -> None:
        """Initialize from a SpecConfig dataclass."""
        self.SeriesID = np.array(config.series_id)
        self.SeriesName = np.array(config.series_name)
        self.Frequency = np.array(config.frequency)
        self.Units = np.array(config.units)
        self.Transformation = np.array(config.transformation)
        self.Category = np.array(config.category)
        self.Blocks = config.blocks_matrix
        self.BlockNames = config.block_names

        if not (self.Blocks[:, 0] == 1).all():
            raise ValueError("All variables must load on global block (column 0 must be 1).")

        self._set_transformations()


    def _init_from_file(self, filename: str) -> None:
        """Initialize from an Excel/CSV file."""
        if filename.lower().endswith(".csv"):
            raw = pd.read_csv(filename)
        else:
            raw = pd.read_excel(filename)

        raw.columns = [str(i).replace(" ", "") for i in raw.columns]
        raw = raw[raw["Model"] == 1].reset_index(drop=True)

        # Sort by frequency
        frequency_order = ["d", "w", "m", "q", "sa", "a"]
        permutations = []
        for freq in frequency_order:
            permutations += list(raw[raw.Frequency == freq].index)
        raw = raw.loc[permutations, :]

        required_fields = ["SeriesID", "SeriesName", "Frequency", "Units", "Transformation", "Category"]
        for field in required_fields:
            if field in raw.columns:
                setattr(self, f"_{field}", raw[field].to_numpy(copy=True))   
                print(f"setting field:  {field} as  {raw[field]}")
            else:
                raise ValueError(f"Missing required column: {field}")

        # Parse blocks
        block_cols = [col for col in raw.columns if "Block" in col]
        Blocks = raw[block_cols].copy()
        Blocks[Blocks.isna()] = 0
        if not (Blocks.iloc[:, 0] == 1).all():
            raise ValueError("All variables must load on global block.")
        self._Blocks = Blocks.to_numpy(copy=True)   
        self._BlockNames = [re.sub(r"Block\d+-", "", col) for col in block_cols]  

        self._set_transformations() 
        
    def _set_transformations(self, verbose: bool = False) -> None:
        """Set human-readable transformation descriptions."""
        from dfm_sp.sp_transformations import MacroTransformations
        self.UnitsTransformed = np.array(
            [MacroTransformations.get_description(t) for t in self.Transformation]
        )

        if verbose:
            print("\nTable 1: Model specification\n")
            print(
                pd.DataFrame({
                    "SeriesID": self.SeriesID,
                    "SeriesName": self.SeriesName,
                    "Units": self.Units,
                    "UnitsTransformed": self.UnitsTransformed,
                })
            )


# Backward-compatible alias — existing code using `load_spec` continues to work.
load_spec = LoadSpec
