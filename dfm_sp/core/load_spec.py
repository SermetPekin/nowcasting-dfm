"""
BSD 3-Clause License

Copyright (c) 2018, Federal Reserve Bank of New York (original MATLAB implementation by Eric Qian and Brandyn Bok)
Copyright (c) 2019, Galib Khan (independent Python translation, not affiliated with FRBNY, https://github.com/MajesticKhan/Nowcasting-Python)

"""

# -------------------------------------------------Libraries
import pandas as pd
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Union


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


# -------------------------------------------------load_spec class
class load_spec:
    # loadSpec Load model specification for a dynamic factor model (DFM)
    #
    # Description:
    #
    #   Load model specification  'Spec' from a Microsoft Excel workbook file
    #   given by 'filename'.
    #
    # Input Arguments:
    #
    #   filename -
    #
    # Output Arguments:
    #
    # spec - 1 x 1 structure with the following fields:
    #     . series_id
    #     . name
    #     . frequency
    #     . units
    #     . transformation
    #     . category
    #     . blocks
    #     . BlockNames
    """
    Python Version Notes:
    spec is a dictionary containing the fields:
        . series_id
        . name
        . frequency
        . units
        . transformation
        . category
        . blocks
        . BlockNames
    """

    def __init__(self, spec_input: Union[str, Path, SpecConfig]):
        if isinstance(spec_input, (str, Path)):
            self._init_from_file(str(spec_input))
        elif isinstance(spec_input, SpecConfig):
            self._init_from_config(spec_input)
        else:
            raise TypeError(
                "spec_input must be a file path strictly pointing to Excel/CSV or a SpecConfig instance"
            )

    def _init_from_config(self, config: SpecConfig):
        # Programmatic init without excel
        self.SeriesID = np.array(config.series_id)
        self.SeriesName = np.array(config.series_name)
        self.Frequency = np.array(config.frequency)
        self.Units = np.array(config.units)
        self.Transformation = np.array(config.transformation)
        self.Category = np.array(config.category)

        self.Blocks = config.blocks_matrix
        self.BlockNames = config.block_names

        if not (self.Blocks[:, 0] == 1).all():
            raise ValueError(
                "All variables must load on global block (column 0 must be 1)."
            )

        self._set_transformations()

    def _init_from_file(self, filename: str):
        # Find and drop series from Spec that are not in Model
        if filename.lower().endswith(".csv"):
            raw = pd.read_csv(filename)
        else:
            raw = pd.read_excel(filename)

        raw.columns = [str(i).replace(" ", "") for i in raw.columns]
        raw = raw[raw["Model"] == 1].reset_index(drop=True)
        # Sort all fields of 'Spec' in order of decreasing frequency
        frequency = ["d", "w", "m", "q", "sa", "a"]
        permutations = []
        for freq in frequency:
            permutations += list(raw[raw.Frequency == freq].index)
        raw = raw.loc[permutations, :]
        # Parse fields given by column names in Excel worksheet
        fldnms = [
            "SeriesID",
            "SeriesName",
            "Frequency",
            "Units",
            "Transformation",
            "Category",
        ]
        for field in fldnms:
            if field in raw.columns:
                setattr(self, field, raw[field].to_numpy(copy=True))
            else:
                raise ValueError(f"{field} column missing from model specification.")
        # Parse blocks
        jColBlock = list(raw.columns[raw.columns.str.contains("Block", case=False)])
        Blocks = raw[jColBlock].copy()
        Blocks[Blocks.isna()] = 0
        if not (Blocks.iloc[:, 0] == 1).all():
            raise ValueError("All variables must load on global block.")
        else:
            self.Blocks = Blocks.to_numpy(copy=True)
        self.BlockNames = [re.sub("Block[0-9]+-", "", i) for i in jColBlock]

        self._set_transformations()

    def _set_transformations(self):
        # Transformations via central registry
        from dfm_sp.sp_transformations import MacroTransformations

        self.UnitsTransformed = np.array(
            [MacroTransformations.get_description(i) for i in self.Transformation]
        )
        # Summarize model specification
        print("\n Table 1: Model specification \n")
        print(
            pd.DataFrame(
                {
                    "SeriesID": self.SeriesID,
                    "SeriesName": self.SeriesName,
                    "Units": self.Units,
                    "UnitsTransformed": self.UnitsTransformed,
                }
            )
        )
