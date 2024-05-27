import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from src.preprocessing.clean_dataset import clean_data

logger = logging.getLogger(__name__)


def preprocess_data(
    file_path: str,
    output_path: str,
    input_headers: list[str],
    pollutant_headers: list[str],
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Loads and preprocesses the dataset by converting date columns, encoding categorical features,
    and creating separate output columns for each pollutant. Saves the preprocessed data to a new file.

    :param file_path: Path to the original dataset file.
    :param output_path: Path to save the preprocessed dataset file.
    :param input_headers: List of input columns to keep.
    :param pollutant_headers: List of pollutant columns to create.
    :return: Tuple of the preprocessed dataset, state label encoder, and city label encoder.
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        logger.info(f"Initial columns in the dataset: {list(data.columns)}")
        logger.info(f"Initial number of rows in the dataset: {len(data)}")

        # Convert column names to lowercase snake case
        data.columns = data.columns.str.lower().str.replace(" ", "_")
        logger.info(
            f"Columns after conversion to lowercase snake case: {list(data.columns)}"
        )

        # Select and rename relevant columns for pollutants
        data = data.rename(
            columns={
                "date_local": "date",
                "no2_mean": "no2_mean",
                "o3_mean": "o3_mean",
                "so2_mean": "so2_mean",
                "co_mean": "co_mean",
            }
        )
        logger.info(f"Columns after renaming: {list(data.columns)}")

        # Filter to include only necessary columns
        data = data[
            ["date", "city", "state", "no2_mean", "o3_mean", "so2_mean", "co_mean"]
        ]
        logger.info(f"Filtered columns: {list(data.columns)}")

        # Convert 'date' column to datetime
        data["date"] = pd.to_datetime(data["date"])
        logger.info("Converted 'date' column to datetime")

        # Create additional time series features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        data['week_of_year'] = data['date'].dt.isocalendar().week
        data['is_weekend'] = data['day_of_week'] >= 5
        logger.info("Created additional time series features: year, month, day_of_week, week_of_year, is_weekend")

        # Encode categorical features
        state_mapper = LabelEncoder()
        city_mapper = LabelEncoder()
        data["state"] = state_mapper.fit_transform(data["state"])
        data["city"] = city_mapper.fit_transform(data["city"])
        logger.info("Encoded 'state' and 'city' columns")

        # Normalize pollutant columns
        for pollutant in pollutant_headers:
            if pollutant in data.columns:
                mean = data[pollutant].mean()
                std = data[pollutant].std()
                data[pollutant] = (data[pollutant] - mean) / std
                logger.info(
                    f"Normalized {pollutant} values with mean: {mean}, std: {std}"
                )

        # Columns to keep
        columns_to_keep = input_headers + pollutant_headers
        logger.info(f"Columns to keep: {columns_to_keep}")

        # Save the intermediate data to the new file
        data.to_csv(output_path, index=False)
        logger.info(f"Intermediate data saved to {output_path}")

        # Clean the dataset
        clean_data(output_path, columns_to_keep)

        # Reload the cleaned data
        cleaned_data = pd.read_csv(output_path)
        logger.info(
            f"Final columns in the cleaned dataset: {list(cleaned_data.columns)}"
        )
        logger.info(f"Final number of rows in the cleaned dataset: {len(cleaned_data)}")

        logger.info(f"Data preprocessed successfully and saved to {output_path}")
        return cleaned_data, state_mapper, city_mapper
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise


if __name__ == "__main__":
    # Example input headers
    input_headers = ["year", "month", "state", "city"]
    # Example pollutant headers
    pollutant_headers = ["no2_mean", "o3_mean", "so2_mean", "co_mean"]

    # Example file paths
    file_path = "data/us_pollution_data.csv"
    output_path = "data/us_pollution_data_preprocessed.csv"

    preprocess_data(file_path, output_path, input_headers, pollutant_headers)
