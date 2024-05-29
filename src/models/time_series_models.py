import logging
from itertools import product
from typing import Dict, List, Tuple, Any, Union
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
from src.utils.utils import filter_data_by_cities

# Setup logging
logger = logging.getLogger(__name__)

ModelConfig = Dict[str, Union[str, Dict[str, Any]]]

warnings.filterwarnings("ignore")


def adf_test(series):
    result = adfuller(series)
    return result[1]  # return p-value


def difference_series(series):
    return series.diff().dropna()


def grid_search_arima(
    y_train, p_range: range, d_range: range, q_range: range
) -> Tuple[Tuple[int, int, int], Any]:
    best_aic = np.inf
    best_order = None
    best_mdl = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    temp_mdl = ARIMA(y_train, order=(p, d, q)).fit()
                    temp_aic = temp_mdl.aic
                    if temp_aic < best_aic:
                        best_aic = temp_aic
                        best_order = (p, d, q)
                        best_mdl = temp_mdl
                except:
                    continue

    return best_order, best_mdl


def grid_search_sarima(
    y_train,
    p_range: range,
    d_range: range,
    q_range: range,
    sp_range: range,
    sd_range: range,
    sq_range: range,
    s_periods: List[int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], Any]:
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_mdl = None

    parameter_combinations = product(
        p_range, d_range, q_range, sp_range, sd_range, sq_range, s_periods
    )

    for params in parameter_combinations:
        p, d, q, sp, sd, sq, s_period = params
        try:
            temp_mdl = SARIMAX(
                y_train, order=(p, d, q), seasonal_order=(sp, sd, sq, s_period)
            ).fit()
            temp_aic = temp_mdl.aic
            if temp_aic < best_aic:
                best_aic = temp_aic
                best_order = (p, d, q)
                best_seasonal_order = (sp, sd, sq, s_period)
                best_mdl = temp_mdl
        except Exception as e:
            print(f"An error occurred for parameters {params}: {e}")
            continue

    return best_order, best_seasonal_order, best_mdl


def train_time_series_model(
    model_type: str,
    y_train: pd.Series,
    hyperparameters: Dict[str, Any],
    s_periods: List[int] = [12],
) -> Any:
    # Change frequency to daily
    y_train = y_train.asfreq("D")

    # Interpolate missing values
    y_train = y_train.interpolate()

    # Check stationarity and difference the series if necessary
    p_value = adf_test(y_train)
    if p_value > 0.05:
        y_train = difference_series(y_train)

    p_range: range = range(*hyperparameters.get("p_range", [0, 4]))
    d_range: range = range(*hyperparameters.get("d_range", [0, 3]))
    q_range: range = range(*hyperparameters.get("q_range", [0, 4]))
    sp_range: range = range(*hyperparameters.get("sp_range", [0, 3]))
    sd_range: range = range(*hyperparameters.get("sd_range", [0, 2]))
    sq_range: range = range(*hyperparameters.get("sq_range", [0, 3]))

    if model_type == "arima":
        # Use grid search to find the best parameters
        order, model = grid_search_arima(y_train, p_range, d_range, q_range)
        logger.info(f"Best ARIMA order found: {order}")
    elif model_type == "sarima":
        # Use grid search to find the best parameters
        order, seasonal_order, model = grid_search_sarima(
            y_train, p_range, d_range, q_range, sp_range, sd_range, sq_range, s_periods
        )
        logger.info(
            f"Best SARIMA order found: {order} with seasonal order {seasonal_order}"
        )
    else:
        raise ValueError(f"Unsupported model type for time series: {model_type}")

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
