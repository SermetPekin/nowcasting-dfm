"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import pandas as pd
import re
from pathlib import Path


def excel_to_csv_spec(excel_path: str, output_csv: str = None):
    """
    Reads an existing DFM Spec Excel file and safely converts it to a standard CSV file.
    If output_csv is not provided, it saves it in the same location with a .csv extension.
    """
    if output_csv is None:
        p = Path(excel_path)
        output_csv = str(p.with_suffix(".csv"))

    try:
        raw = pd.read_excel(excel_path)
        # We don't filter columns or drop ignored models here because we want
        # the CSV to act exactly like the full workspace spreadsheet for the user.
        raw.to_csv(output_csv, index=False)
        print(f"✅ Successfully converted Excel Spec to CSV: {output_csv}")
        return output_csv
    except Exception as e:
        print(f"❌ Failed to convert {excel_path} to CSV. Error: {e}")
        return None


def excel_to_python_spec(
    excel_path: str, output_script: str = "generated_spec_config.py"
):
    """
    Reads an existing DFM Spec Excel file and generates a Python
    script declaring its identical SpecConfig programmatic structure.
    This helps migrate classical users into the modern programmatic infrastructure.
    """
    raw = pd.read_excel(excel_path)
    # clean spaces
    raw.columns = [str(c).replace(" ", "") for c in raw.columns]

    # Filter only applied model ones
    raw = raw[raw["Model"] == 1].reset_index(drop=True)

    # Frequency sort exact match as load_spec
    frequency = ["d", "w", "m", "q", "sa", "a"]
    permutations = []
    for freq in frequency:
        permutations += list(raw[raw.Frequency == freq].index)
    raw = raw.loc[permutations, :]

    # Extract
    series_ids = raw["SeriesID"].tolist()
    series_names = raw["SeriesName"].tolist()
    freqs = raw["Frequency"].tolist()
    units = raw["Units"].tolist()
    trans = raw["Transformation"].tolist()
    cats = raw["Category"].tolist()

    # Extract blocks
    jColBlock = list(raw.columns[raw.columns.str.contains("Block", case=False)])
    Blocks = raw[jColBlock].copy().fillna(0).astype(int)
    block_matrix = Blocks.values.tolist()
    block_names = [re.sub("Block[0-9]+-", "", str(i)) for i in jColBlock]

    # Generate python code
    python_code = f"""import numpy as np
from dfm_sp import SpecConfig

# Auto-Generated from {Path(excel_path).name}
def get_custom_config() -> SpecConfig:
    return SpecConfig(
        series_id={series_ids},
        series_name={series_names},
        frequency={freqs},
        units={units},
        transformation={trans},
        category={cats},
        block_names={block_names},
        blocks_matrix=np.array({block_matrix})
    )
"""

    with open(output_script, "w") as f:
        f.write(python_code)

    print(f"Generated python programmatic spec at: {output_script}")
    return python_code
