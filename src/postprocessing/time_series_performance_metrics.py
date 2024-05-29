import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
import numpy as np
import pandas as pd

from src.utils.utils import filter_data_by_cities

logger = logging.getLogger(__name__)


def mase(y_true, y_pred, y_train, seasonality=1):
    n = len(y_train)
    d = np.abs(np.diff(y_train, n=seasonality)).sum() / (n - seasonality)
    if d == 0:  # Prevent division by zero
        d = np.finfo(float).eps  # Use a small epsilon value
    errors = np.abs(y_true - y_pred)
    mase_value = errors.mean() / d
    return mase_value


def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Prevent division by zero by adding a small value to the denominator
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    smape_value = 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)
    return smape_value


def evaluate_time_series_models(
    models, test_data, train_data, pollutant_headers, seasonality=1
):
    metrics = {pollutant: [] for pollutant in pollutant_headers}

    for model_type, pollutant, state, city, model in models:
        logger.info(
            f"Evaluating time series model for {pollutant} ({model_type}) in {city}, {state}"
        )

        specific_cities = [{"state": state, "city": city}]
        city_test_data = filter_data_by_cities(test_data, specific_cities)
        city_train_data = filter_data_by_cities(train_data, specific_cities)

        if city_test_data.empty or city_train_data.empty:
            logger.warning(f"No data found for city: {city}, state: {state}")
            continue

        city_test_data["date"] = pd.to_datetime(city_test_data["date"])
        city_test_data.set_index("date", inplace=True)
        city_train_data["date"] = pd.to_datetime(city_train_data["date"])
        city_train_data.set_index("date", inplace=True)

        y_test = city_test_data[pollutant]
        y_train = city_train_data[pollutant]

        if y_test.empty or y_train.empty:
            logger.warning(f"No data available for {pollutant} in {city}, {state}")
            continue

        if isinstance(model, (ARIMAResultsWrapper, SARIMAXResultsWrapper)):
            predictions = model.get_forecast(steps=len(y_test)).predicted_mean
        else:
            raise ValueError(f"Model type {model_type} not supported for time series")

        # Align indices of y_test and predictions
        y_test, predictions = y_test.align(predictions, join="inner")

        if y_test.empty or predictions.empty:
            logger.warning(
                f"No overlapping data for {pollutant} in {city}, {state} after alignment"
            )
            continue

        logger.debug(f"Evaluating model: y_test={y_test}, predictions={predictions}")

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mase_value = mase(y_test, predictions, y_train, seasonality=seasonality)
        smape_value = smape(y_test, predictions)

        metrics[pollutant].append(
            (
                model_type,
                {
                    "state": state,
                    "city": city,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mase": mase_value,
                    "smape": smape_value,
                },
            )
        )
        logger.info(
            f"Evaluation metrics for {pollutant} ({model_type}) in {city}, {state}: MSE = {mse}, MAE = {mae}, R2 = {r2}, MASE = {mase_value}, sMAPE = {smape_value}"
        )

    return metrics


def choose_best_time_series_models(metrics, metric="mse"):

    if not metrics or all(not v for v in metrics.values()):
        return list()

    best_models = {}

    for pollutant, model_metrics in metrics.items():
        best_model = min(model_metrics, key=lambda x: x[1][metric])
        best_models[pollutant] = best_model
        logger.info(
            f"Best time series model for pollutant '{pollutant}':\n"
            f"Location: State = {best_model[1]['state']}, City = {best_model[1]['city']}\n"
            f"Model: {best_model[0]}\n"
            f"Performance Metrics:\n"
            f"  - {metric.upper()}: {best_model[1][metric]}\n"
            f"  - MAE: {best_model[1]['mae']}\n"
            f"  - R2: {best_model[1]['r2']}\n"
            f"  - MASE: {best_model[1]['mase']}\n"
            f"  - sMAPE: {best_model[1]['smape']}"
        )

    return best_models
