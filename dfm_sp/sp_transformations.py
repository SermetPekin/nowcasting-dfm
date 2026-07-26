import numpy as np
import pandas as pd
from typing import Dict, Callable


class MacroTransformations:
    """
    A centralized registry for macroeconomic data transformations used in Nowcasting and DFM estimation.
    This class handles the parsing and mathematical manipulation of variables natively to ensure stationarity
    without exploding variance limits.
    """

    @staticmethod
    def get_description(formula_code: str) -> str:
        """Returns the human-readable description for a given transformation code."""
        descriptions = {
            "lin": "Levels (No Transformation)",
            "chg": "Change (Difference)",
            "ch1": "Year over Year Change (Difference)",
            "pch": "Percent Change",
            "pc1": "Year over Year Percent Change",
            "pca": "Percent Change (Annual Rate)",
            "cch": "Continuously Compounded Rate of Change",
            "cca": "Continuously Compounded Annual Rate of Change",
            "log": "Natural Log",
            "dln": "Log Difference (First diff of log)",
            "dl1": "Year over Year Log Difference",
            "d2l": "Second Log Difference",
            "zsc": "Z-Score Standardization",
        }
        return descriptions.get(formula_code, f"Unknown ({formula_code})")

    @staticmethod
    def get_formulas(
        t1: int, step: int, n: float
    ) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        """
        Returns a dictionary of executable lambda functions properly scoped to the
        target time-step arrays.

        Args:
            t1: Base offset index (typically step - 1)
            step: Timestep jump (1 for monthly, 3 for quarterly mapping over months)
            n: Period-years offset (step / 12)
        """
        return {
            "lin": lambda x: x,
            "chg": lambda x: np.append(
                np.nan, x[t1 + step :: step] - x[t1 : -1 - t1 : step]
            ),
            "ch1": lambda x: x[12 + t1 :: step] - x[t1:-12:step],
            "pch": lambda x: (
                np.append(np.nan, x[t1 + step :: step] / x[t1 : -1 - t1 : step]) - 1
            )
            * 100,
            "pc1": lambda x: ((x[12 + t1 :: step] / x[t1:-12:step]) - 1) * 100,
            "pca": lambda x: (
                np.append(np.nan, x[t1 + step :: step] / x[t1:-step:step]) ** (1 / n)
                - 1
            )
            * 100,
            "log": lambda x: np.log(x),
            "dln": lambda x: np.append(
                np.nan, np.log(x[t1 + step :: step]) - np.log(x[t1 : -1 - t1 : step])
            )
            * 100,
            "dl1": lambda x: (np.log(x[12 + t1 :: step]) - np.log(x[t1:-12:step]))
            * 100,
            "d2l": lambda x: np.append(
                np.nan,
                np.append(
                    np.nan,
                    np.log(x[t1 + step + step :: step])
                    - np.log(x[t1 + step : -1 - step : step])
                    - (
                        np.log(x[t1 + step : -1 - step : step])
                        - np.log(x[t1 : -1 - step - step : step])
                    ),
                ),
            )
            * 100,
            "zsc": lambda x: (x - np.nanmean(x)) / np.nanstd(x),
        }
