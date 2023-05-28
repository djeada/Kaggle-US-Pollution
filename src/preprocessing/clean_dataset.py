import numpy as np
import pandas as pd


def clean_dataset(data_frame: pd.DataFrame, columns_to_keep) -> pd.DataFrame:
    """
    Cleans the dataset.

    :param data_frame: The dataset to clean.
    :return: The cleaned dataset.
    """
    # drop unnecessary columns and keep only relevant columns
    data_frame = data_frame[columns_to_keep]

    # Convert all features to numeric
    for column_name in data_frame.columns:
        if data_frame[column_name].dtype == object:
            data_frame[column_name] = pd.to_numeric(
                data_frame[column_name], errors="coerce"
            )

    # Fill nan with mean
    data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
    data_frame = data_frame.fillna(
        data_frame.mean() if data_frame.mean().notnull().all() else 0
    )

    return data_frame
