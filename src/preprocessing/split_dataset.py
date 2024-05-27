# split_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def split_dataset(file_path, test_size=0.2, random_state=42):
    try:
        data = pd.read_csv(file_path)
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
