from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    train_x: pd.DataFrame
    train_y: pd.DataFrame
    test_x: pd.DataFrame
    test_y: pd.DataFrame


def split_dataset(
    x_data_frame: pd.DataFrame,
    y_data_frame: pd.DataFrame,
    split_ratio: float = 1 / 3,
    save_to_file: bool = False,
) -> DatasetSplit:
    """
    Splits the data frame into a train and test set.
    :param data_frame: The data frame to split.
    :param split_ratio: The ratio of the test set.
    :param save_to_file: Whether to save the data frames to file.
    :return: The train and test data frames.
    """

    train_test_data = train_test_split(
        x_data_frame, y_data_frame, test_size=split_ratio, random_state=85
    )

    train_x = train_test_data[0]
    train_y = train_test_data[2]

    test_x = train_test_data[1]
    test_y = train_test_data[3]

    if save_to_file:
        train_x.to_csv("../output/train_x.csv")
        train_y.to_csv("../output/train_y.csv")
        test_x.to_csv("../output/test_x.csv")
        test_y.to_csv("../output/test_y.csv")

    return DatasetSplit(train_x, train_y, test_x, test_y)


def load_dataset_split() -> DatasetSplit:
    """
    Loads the train and test data frames from file.
    :return: The train and test data frames.
    """
    train_x = pd.read_csv("../output/train_x.csv")
    train_y = pd.read_csv("../output/train_y.csv")
    test_x = pd.read_csv("../output/test_x.csv")
    test_y = pd.read_csv("../output/test_y.csv")

    return DatasetSplit(train_x, train_y, test_x, test_y)
