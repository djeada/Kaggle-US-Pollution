import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def clean_data(
    file_path: str, columns_to_keep: list[str], convert_to_numeric: bool = True
) -> (pd.DataFrame, dict):
    data = pd.read_csv(file_path)
    logger.info(f"Initial columns in the dataset: {list(data.columns)}")
    logger.info(f"Initial number of rows in the dataset: {len(data)}")

    data = data[columns_to_keep]
    logger.info(f"Columns after keeping specified ones: {list(data.columns)}")
    logger.info(f"Number of rows after filtering columns: {len(data)}")

    # Ensure the 'date' column is datetime and not converted to numeric
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        logger.info("Converted 'date' column to datetime")

    encoders = {}

    if convert_to_numeric:
        # Label encode categorical features and convert them to numeric
        for col in data.select_dtypes(include=["object"]).columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col].astype(str))
            encoders[col] = encoder
            logger.info(f"Encoded column '{col}' with LabelEncoder")

    # Replace infinite values with NaN in numeric columns
    numeric_cols = data.select_dtypes(include="number").columns
    data[numeric_cols] = data[numeric_cols].replace(
        [float("inf"), float("-inf")], float("nan")
    )
    num_infs = data[numeric_cols].isin([float("inf"), float("-inf")]).sum().sum()
    logger.info(f"Replaced {num_infs} infinite values with NaN in numeric columns")

    # Fill NaNs with column means in numeric columns
    num_nans_before = data[numeric_cols].isna().sum().sum()
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    num_nans_after = data[numeric_cols].isna().sum().sum()
    logger.info(
        f"Filled NaN values with column mean in numeric columns: {num_nans_before} NaNs before, {num_nans_after} NaNs after"
    )

    # Drop remaining NaNs if any
    data.dropna(inplace=True)
    logger.info(f"Dropped rows with remaining NaN values: {len(data)} rows left")

    # Drop duplicate rows
    data.drop_duplicates(inplace=True)
    logger.info(f"Dropped duplicate rows: {len(data)} rows left")

    data.to_csv(file_path, index=False)
    logger.info(f"Data cleaned successfully and saved to {file_path}")

    return data, encoders


if __name__ == "__main__":
    # Example columns to keep
    columns_to_keep = ["date", "city", "state", "pollutant", "value"]

    # Example file path
    file_path = "data/us_pollution_data.csv"

    clean_data(file_path, columns_to_keep)
