import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.clean_dataset import clean_data

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    logger.info(f"Initial columns in the dataset: {list(data.columns)}")
    logger.info(f"Initial number of rows in the dataset: {len(data)}")
    return data


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    logger.info(
        f"Columns after conversion to lowercase snake case: {list(data.columns)}"
    )
    return data


def rename_columns(data: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    data = data.rename(
        columns={k: v for k, v in rename_dict.items() if k in data.columns}
    )
    logger.info(f"Columns after renaming: {list(data.columns)}")
    return data


def filter_columns(data: pd.DataFrame, columns_to_select: list[str]) -> pd.DataFrame:
    data = data[columns_to_select]
    logger.info(f"Filtered columns: {list(data.columns)}")
    return data


def convert_to_datetime(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    data[date_column] = pd.to_datetime(data[date_column])
    logger.info("Converted 'date' column to datetime")
    return data


def create_time_features(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    data["year"] = data[date_column].dt.year
    data["month"] = data[date_column].dt.month
    data["day"] = data[date_column].dt.day
    data["day_of_week"] = data[date_column].dt.dayofweek
    data["week_of_year"] = data[date_column].dt.isocalendar().week
    data["is_weekend"] = data["day_of_week"] >= 5
    logger.info(
        "Created additional time series features: year, month, day_of_week, week_of_year, is_weekend"
    )
    return data


def normalize_pollutant_columns(
    data: pd.DataFrame, pollutant_headers: list[str]
) -> pd.DataFrame:
    for pollutant in pollutant_headers:
        if pollutant in data.columns:
            mean = data[pollutant].mean()
            std = data[pollutant].std()
            data[pollutant] = (data[pollutant] - mean) / std
            logger.info(f"Normalized {pollutant} values with mean: {mean}, std: {std}")
    return data


def squash_duplicates(data: pd.DataFrame, group_by_columns: list[str]) -> pd.DataFrame:
    data = data.groupby(group_by_columns, as_index=False).mean()
    logger.info("Squashed duplicated rows by averaging pollutant values")
    return data


def save_data(data: pd.DataFrame, output_path: str):
    data.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}")


def preprocess_data(
    file_path: str,
    output_path_regression: str,
    output_path_timeseries: str,
    input_headers: list[str],
    pollutant_headers: list[str],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    try:
        data = load_data(file_path)
        data = clean_column_names(data)
        rename_dict = {
            "date_local": "date",
            "no2_mean": "no2_mean",
            "o3_mean": "o3_mean",
            "so2_mean": "so2_mean",
            "co_mean": "co_mean",
        }
        data = rename_columns(data, rename_dict)
        columns_to_select = ["date", "city", "state"] + pollutant_headers
        data = filter_columns(data, columns_to_select)
        logger.info(f"Number of rows after filtering columns: {len(data)}")

        data = convert_to_datetime(data, "date")
        data = create_time_features(data, "date")
        data = normalize_pollutant_columns(data, pollutant_headers)
        logger.info(f"Number of rows after normalizing pollutant columns: {len(data)}")

        group_by_columns = [
            "date",
            "city",
            "state",
            "year",
            "month",
            "day",
            "day_of_week",
            "week_of_year",
            "is_weekend",
        ]
        data = squash_duplicates(data, group_by_columns)
        logger.info(f"Number of rows after squashing duplicates: {len(data)}")

        # Create and save time series dataset
        time_series_columns = columns_to_select + [
            "year",
            "month",
            "day",
            "day_of_week",
            "week_of_year",
            "is_weekend",
        ]
        time_series_data = filter_columns(data, time_series_columns)
        save_data(time_series_data, output_path_timeseries)
        time_series_data, _ = clean_data(
            output_path_timeseries, time_series_columns, convert_to_numeric=False
        )
        time_series_data = pd.read_csv(output_path_timeseries)
        logger.info(f"Time series data saved to {output_path_timeseries}")
        logger.info(f"Number of rows in time series data: {len(time_series_data)}")

        # Create and save regression dataset
        regression_columns_to_keep = input_headers + pollutant_headers
        regression_data = filter_columns(data, regression_columns_to_keep)
        save_data(regression_data, output_path_regression)
        regression_data, encoders = clean_data(
            output_path_regression, regression_columns_to_keep
        )
        regression_data = pd.read_csv(output_path_regression)
        logger.info(f"Regression data saved to {output_path_regression}")
        logger.info(f"Number of rows in regression data: {len(regression_data)}")

        logger.info("Data preprocessed successfully")

        return time_series_data, regression_data, encoders
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise
