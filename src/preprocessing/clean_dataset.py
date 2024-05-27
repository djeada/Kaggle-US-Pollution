from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def clean_data(file_path: str, columns_to_keep: list[str]) -> None:
    """
    Cleans the dataset file at the given path and saves it back to the same path.
    The cleaning operations include dropping missing values, duplicates, and keeping specified columns.

    :param file_path: Path to the dataset file.
    :param columns_to_keep: List of columns to keep in the dataset.
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Initial columns in the dataset: {list(data.columns)}")
        logger.info(f"Initial number of rows in the dataset: {len(data)}")

        # Check if columns_to_keep are in the data columns
        missing_columns = [col for col in columns_to_keep if col not in data.columns]
        if missing_columns:
            logger.error(f"Columns to keep not found in dataset: {missing_columns}")
            raise KeyError(f"Columns to keep not found in dataset: {missing_columns}")

        # Keep only relevant columns
        data = data[columns_to_keep]
        logger.info(f"Columns after keeping specified ones: {list(data.columns)}")

        # Convert all features to numeric
        data = data.apply(pd.to_numeric, errors="coerce")
        logger.info("Converted all features to numeric")

        # Replace infinite values with NaN
        num_infs = np.isinf(data).sum().sum()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Replaced {num_infs} infinite values with NaN")

        # Fill NaN values with column mean
        num_nans_before = data.isna().sum().sum()
        data.fillna(data.mean(), inplace=True)
        num_nans_after = data.isna().sum().sum()
        logger.info(
            f"Filled NaN values with column mean: {num_nans_before} NaNs before, {num_nans_after} NaNs after"
        )

        # Drop remaining NaN values
        initial_length = len(data)
        data.dropna(inplace=True)
        dropped_nans = initial_length - len(data)
        logger.info(
            f"Dropped rows with remaining NaN values: {dropped_nans} rows dropped"
        )

        # Drop duplicate rows
        initial_length = len(data)
        data.drop_duplicates(inplace=True)
        dropped_duplicates = initial_length - len(data)
        logger.info(f"Dropped duplicate rows: {dropped_duplicates} rows dropped")

        # Save the cleaned data back to the file
        data.to_csv(file_path, index=False)
        logger.info(f"Data cleaned successfully and saved to {file_path}")
        logger.info(f"Final number of rows in the dataset: {len(data)}")
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise


if __name__ == "__main__":
    # Example columns to keep
    columns_to_keep = ["date", "city", "state", "pollutant", "value"]

    # Example file path
    file_path = "data/us_pollution_data.csv"

    clean_data(file_path, columns_to_keep)
