import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_dataset(file_path, test_size=0.2, random_state=42, date_column=None):
    try:
        data = pd.read_csv(file_path)

        if date_column and date_column in data.columns:
            # Convert the date column to datetime if it's not already
            data[date_column] = pd.to_datetime(data[date_column])
            # Sort the data by the date column
            data = data.sort_values(by=date_column)
            # Determine the split index
            split_index = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_index]
            test_data = data.iloc[split_index:]
        else:
            train_data, test_data = train_test_split(
                data, test_size=test_size, random_state=random_state
            )

        train_data.to_csv("data/train_data.csv", index=False)
        test_data.to_csv("data/test_data.csv", index=False)

        logger.info(
            f"Data split into training and testing sets with test size {test_size}"
        )
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise
