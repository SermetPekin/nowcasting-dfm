"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List


class PublicationRule:
    """Base class for modeling specific macroeconomic data release schedules."""

    pass


class FixedDayRule(PublicationRule):
    def __init__(self, day: int, lag: int, weekend: str = "next"):
        """
        Args:
            day: e.g. 15 (Published on the 15th of the month)
            lag: e.g. 1 (The data published represents the previous month)
            weekend: 'next' (shift to Monday), 'prev' (shift to Friday), or None.
        """
        self.day = day
        self.lag = lag
        self.weekend = weekend


class WeekdayRule(PublicationRule):
    def __init__(self, weekday: int, n: int, lag: int):
        """
        Args:
            weekday: 0=Monday, 4=Friday
            n: e.g. 1 (First Friday)
            lag: e.g. 1 (Represents previous month)
        """
        self.weekday = weekday
        self.n = n
        self.lag = lag


def get_nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> datetime:
    """Helper to find 'First Friday' or 'Second Tuesday'."""
    first_day = datetime(year, month, 1)
    first_weekday = first_day.weekday()
    days_to_target = (weekday - first_weekday) % 7
    target_date = first_day + timedelta(days=days_to_target + (n - 1) * 7)
    return target_date


class VintageMaker:
    """
    Synthesizes historical 'Vintage' data frames from a single fully-revised modern dataset
    by systematically blinding data that mathematically could not have been published yet
    on the target date.
    """

    def __init__(self, rules_dict: Dict[str, PublicationRule]):
        """
        rules_dict maps SeriesID strings from the SpecConfig to their theoretical publication rule.
        """
        self.rules_dict = rules_dict

    def _get_publish_date(
        self, reference_date: pd.Timestamp, rule: PublicationRule
    ) -> pd.Timestamp:
        """
        Given the economic *reference* date (e.g., the 'March' data point),
        calculates out the exact future date that point was actually published.
        """
        # The month the report is physically released
        release_month_date = reference_date + pd.DateOffset(months=rule.lag)

        if isinstance(rule, FixedDayRule):
            try:
                # Handle edge cases like Feb 29/30 safely
                publish_date = release_month_date.replace(day=rule.day)
            except ValueError:
                # If day=31 and month has 30 days, clamp to end of month.
                publish_date = release_month_date + pd.offsets.MonthEnd(0)

            dt = pd.Timestamp(publish_date)

            # Handle Weekend Adjustments dynamically
            if rule.weekend and dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
                if rule.weekend == "next":
                    days_to_add = 2 if dt.weekday() == 5 else 1
                    dt += pd.Timedelta(days=days_to_add)
                elif rule.weekend == "prev":
                    days_to_sub = 1 if dt.weekday() == 5 else 2
                    dt -= pd.Timedelta(days=days_to_sub)

            return dt

        elif isinstance(rule, WeekdayRule):
            dt = get_nth_weekday_of_month(
                release_month_date.year, release_month_date.month, rule.weekday, rule.n
            )
            return pd.Timestamp(dt)

        else:
            raise NotImplementedError(f"Unsupported rule type: {type(rule)}")

    def __call__(
        self, full_dataset: pd.DataFrame, target_vintage_date: str
    ) -> pd.DataFrame:
        """Allows the instance to be called directly: maker(df, '2016-12-10')"""
        return self.synthesize_vintage(full_dataset, target_vintage_date)

    def synthesize_vintage(
        self, full_dataset: pd.DataFrame, target_vintage_date: str
    ) -> pd.DataFrame:
        """
        Takes the fully revised modern data and blanks out (NaNs) all observations
        that were published AFTER the target_vintage_date based on the mathematical rules.
        """
        target_timestamp = pd.to_datetime(target_vintage_date)
        vintage_df = full_dataset.copy()

        # Iterate over every series the user provided rules for
        for col in vintage_df.columns:
            if col in self.rules_dict:
                rule = self.rules_dict[col]

                # Check each observation in the column
                for ref_date, value in vintage_df[col].items():
                    if pd.isna(value):
                        continue  # Already missing, ignore it

                    publish_timestamp = self._get_publish_date(
                        pd.Timestamp(ref_date), rule
                    )

                    # If this specific data point was published AFTER the time machine target...
                    if publish_timestamp > target_timestamp:
                        # Blind it out. The econometrician at the time couldn't have seen it.
                        vintage_df.at[ref_date, col] = np.nan
            else:
                # If no rule was provided, we assume strict causality (it's unknown if we are past the reference date)
                # For safety, default to 1 month lag if no rule provided
                for ref_date in vintage_df.index:
                    if pd.Timestamp(ref_date) >= target_timestamp:
                        vintage_df.at[ref_date, col] = np.nan

        return vintage_df
