from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.models.base_model import BaseModel


def calculate_r2_score(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the R^2 score.
    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The R^2 score.
    """
    return r2_score(test_y, predicted_y)


def calculate_rmse(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the RMSE.
    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The RMSE.
    """
    return np.sqrt(mean_squared_error(test_y, predicted_y))


def calculate_nrmse(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the NRMSE.
    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The NRMSE.
    """
    return np.sqrt(mean_squared_error(test_y, predicted_y)) / np.std(predicted_y)


def calculate_mae(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the MAE.
    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The MAE.
    """
    return mean_absolute_error(test_y, predicted_y)
