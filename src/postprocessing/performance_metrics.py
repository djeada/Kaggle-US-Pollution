from typing import Iterable

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def validate_shapes(test_y: np.ndarray, predicted_y: np.ndarray):
    if test_y.shape != predicted_y.shape:
        raise ValueError("Shape of both `test_y` and `predicted_y` must be same")


def calculate_r2_score(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the R^2 score.

    Args:
        test_y (np.ndarray): The test output data.
        predicted_y (np.ndarray): The predicted output data.

    Returns:
        float: The R^2 score.
    """
    validate_shapes(test_y, predicted_y)
    return r2_score(test_y, predicted_y)


def calculate_rmse(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the RMSE.

    Args:
        test_y (np.ndarray): The test output data.
        predicted_y (np.ndarray): The predicted output data.

    Returns:
        float: The RMSE.
    """
    validate_shapes(test_y, predicted_y)
    return np.sqrt(mean_squared_error(test_y, predicted_y))


def calculate_nrmse(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the NRMSE.

    Args:
        test_y (np.ndarray): The test output data.
        predicted_y (np.ndarray): The predicted output data.

    Returns:
        float: The NRMSE.
    """
    validate_shapes(test_y, predicted_y)
    return np.sqrt(mean_squared_error(test_y, predicted_y)) / np.std(predicted_y)


def calculate_mae(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the MAE.

    Args:
        test_y (np.ndarray): The test output data.
        predicted_y (np.ndarray): The predicted output data.

    Returns:
        float: The MAE.
    """
    validate_shapes(test_y, predicted_y)
    return mean_absolute_error(test_y, predicted_y)


def evaluate_models(models, dataset_split):
    scores = []
    for model in models:
        model_name = model.__class__.__name__
        test_y_array = dataset_split.test_y.to_numpy()
        predicted_y_array = model.predict(dataset_split.test_x.to_numpy())
        score = {
            "model": model_name,
            "model_instance": model,
            "r2_score": calculate_r2_score(test_y_array, predicted_y_array),
            "rmse": calculate_rmse(test_y_array, predicted_y_array),
            "nrmse": calculate_nrmse(test_y_array, predicted_y_array),
        }
        scores.append(score)

    best_model = max(scores, key=lambda score: score["r2_score"])
    return best_model, scores
