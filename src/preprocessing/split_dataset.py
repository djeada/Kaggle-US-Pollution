from dataclasses import dataclass
import pandas as pd
from typing import Tuple


@dataclass
class DatasetSplit:
    train_x: pd.DataFrame
    train_y: pd.DataFrame
    test_x: pd.DataFrame
    test_y: pd.DataFrame


def split_dataset(
    x_data: pd.DataFrame,
    y_data: pd.DataFrame,
    split_ratio: float = 0.8,
    save_to_file: bool = False,
) -> DatasetSplit:
    """
    Splits the data frames into a train and test set.
    :param x_data: The input data frame to split.
    :param y_data: The output data frame to split.
    :param split_ratio: The ratio of the training set.
    :param save_to_file: Whether to save the data frames to file.
    :return: The DatasetSplit containing train and test data frames.
    """

    split_point = int(len(x_data) * split_ratio)
    x_train = x_data[:split_point]
    y_train = y_data[:split_point]

    x_test = x_data[split_point:]
    y_test = y_data[split_point:]

    if save_to_file:
        x_train.to_csv("../output/x_train.csv")
        y_train.to_csv("../output/y_train.csv")
        x_test.to_csv("../output/x_test.csv")
        y_test.to_csv("../output/y_test.csv")

    return DatasetSplit(x_train, y_train, x_test, y_test)


def load_dataset_split() -> DatasetSplit:
    """
    Loads the train and test data frames from file.
    :return: The DatasetSplit containing train and test data frames.
    """
    x_train = pd.read_csv(
        "../output/x_train.csv", index_col="Date Local", parse_dates=True
    )
    y_train = pd.read_csv(
        "../output/y_train.csv", index_col="Date Local", parse_dates=True
    )
    x_test = pd.read_csv(
        "../output/x_test.csv", index_col="Date Local", parse_dates=True
    )
    y_test = pd.read_csv(
        "../output/y_test.csv", index_col="Date Local", parse_dates=True
    )

    return DatasetSplit(x_train, y_train, x_test, y_test)
