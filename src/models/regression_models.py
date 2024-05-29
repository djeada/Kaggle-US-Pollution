import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import xgboost as xgb  # Importing XGBoost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModelConfig = Dict[str, Union[str, Dict[str, Any]]]


def train_single_regression_model(
    pollutant: str, X_train: pd.DataFrame, y_train: pd.Series, model_config: ModelConfig
) -> Tuple[str, str, Any]:
    model_type = model_config["type"]
    hyperparameters = model_config.get("hyperparameters", {})

    logger.info(
        f"Training regression model for {pollutant} with type {model_type} and hyperparameters {hyperparameters}"
    )

    if model_type == "random_forest":
        model = RandomForestRegressor(**hyperparameters)
    elif model_type == "linear_regression":
        model = LinearRegression(**hyperparameters)
    elif model_type == "mlp":
        model = MLPRegressor(**hyperparameters)
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(**hyperparameters)
    else:
        raise ValueError(f"Unsupported model type for regression: {model_type}")

    model.fit(X_train, y_train)
    logger.info(f"Regression model for {pollutant} ({model_type}) trained successfully")
    return model_type, pollutant, model


def train_regression_models(
    train_data: pd.DataFrame,
    input_headers: List[str],
    pollutant_headers: List[str],
    model_configs: List[ModelConfig],
    timeout: int = 300,  # Default timeout set to 5 minutes
) -> Dict[str, List[Tuple[str, Any]]]:
    models = {pollutant: [] for pollutant in pollutant_headers}

    logger.info("Starting training of regression models")
    logger.info(f"Model configurations provided: {model_configs}")

    tasks = [
        (
            pollutant,
            train_data[input_headers],
            train_data[pollutant],
            model_config,
        )
        for pollutant in pollutant_headers
        for model_config in model_configs
    ]

    with ProcessPoolExecutor() as executor:
        future_to_task = {
            executor.submit(train_single_regression_model, *task): task
            for task in tasks
        }

        for future in future_to_task:
            try:
                model_type, pollutant, model = future.result(timeout=timeout)
                models[pollutant].append((model_type, model))
            except TimeoutError:
                logger.warning(f"Training timed out for task: {future_to_task[future]}")
            except Exception as e:
                logger.error(f"Error occurred for task: {future_to_task[future]} - {e}")

    logger.info("Completed training of regression models")
    return models
