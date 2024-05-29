import logging
from typing import Dict, List, Tuple, Any, Union
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils.utils import filter_data_by_cities

# Setup logging
logger = logging.getLogger(__name__)

ModelConfig = Dict[str, Union[str, Dict[str, Any]]]


def train_time_series_model(
    model_type: str, y_train: pd.Series, hyperparameters: Dict[str, Any]
) -> Any:
    # Change frequency to daily
    y_train = y_train.asfreq("D")

    # Interpolate missing values
    y_train = y_train.interpolate()

    if model_type == "arima":
        order = hyperparameters.get("order", (1, 1, 1))
        model = ARIMA(y_train, order=order).fit()
    elif model_type == "sarima":
        order = hyperparameters.get("order", (1, 1, 1))
        seasonal_order = hyperparameters.get("seasonal_order", (1, 1, 1, 12))
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit()
    else:
        raise ValueError(f"Unsupported model type for time series: {model_type}")

    logger.info(
        f"Time series model {model_type} fitted with order {order} and seasonal order {hyperparameters.get('seasonal_order')}"
    )
    return model


def train_time_series_model_wrapper(
    pollutant: str, train_data: pd.DataFrame, model_config: ModelConfig
) -> List[Tuple[str, str, str, str, Any]]:
    specific_cities = model_config.get("specific_cities", [])
    if not specific_cities:
        logger.error("Specific cities must be specified for time series models")
        raise ValueError("Specific cities must be specified for time series models")

    models = []
    for city in specific_cities:
        city_train_data = filter_data_by_cities(train_data, [city])
        if city_train_data.empty:
            logger.warning(
                f"No data found for city: {city['city']}, state: {city['state']}"
            )
            continue

        city_train_data["date"] = pd.to_datetime(city_train_data["date"])
        city_train_data.set_index("date", inplace=True)
        y_train = city_train_data[pollutant]

        model_type = model_config["type"]
        hyperparameters = model_config.get("hyperparameters", {})

        logger.info(
            f"Training time series model for {pollutant} in {city['city']}, {city['state']} with type {model_type} and hyperparameters {hyperparameters}"
        )

        try:
            model = train_time_series_model(model_type, y_train, hyperparameters)

            logger.info(
                f"Time series model for {pollutant} in {city['city']}, {city['state']} ({model_type}) trained successfully"
            )
            models.append((model_type, pollutant, city["state"], city["city"], model))
        except Exception as e:
            logger.error(
                f"Failed to train model for {city['city']}, {city['state']}: {e}"
            )

    return models


def train_time_series_models(
    train_data: pd.DataFrame,
    pollutant_headers: List[str],
    model_configs: List[ModelConfig],
    timeout: int = 300,
) -> List[Tuple[str, str, str, str, Any]]:
    required_columns = {"date", "state", "city"}
    if not required_columns.issubset(train_data.columns):
        missing_columns = required_columns - set(train_data.columns)
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    logger.info("Starting training of time series models")
    logger.info(f"Model configurations provided: {model_configs}")

    tasks = [
        (pollutant, train_data, model_config)
        for model_config in model_configs
        for pollutant in pollutant_headers
    ]

    models = []
    with ProcessPoolExecutor() as executor:
        future_to_task = {
            executor.submit(train_time_series_model_wrapper, *task): task
            for task in tasks
        }

        for future in future_to_task:
            try:
                result = future.result(timeout=timeout)
                models.extend(result)
            except TimeoutError:
                logger.warning(f"Training timed out for task: {future_to_task[future]}")
            except Exception as e:
                logger.error(f"Error occurred for task: {future_to_task[future]} - {e}")

    logger.info("Completed training of time series models")
    return models
