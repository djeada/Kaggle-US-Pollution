import logging
from pathlib import Path

import yaml

from config.logging_config import setup_logging
from src.models.regression_models import train_regression_models
from src.models.time_series_models import train_time_series_models
from src.postprocessing.plot_regression_prediction import \
    plot_regression_predictions
from src.postprocessing.plot_time_series_prediction import \
    plot_time_series_predictions
from src.postprocessing.regression_performance_metrics import (
    choose_best_regression_models, evaluate_regression_models)
from src.postprocessing.time_series_performance_metrics import (
    choose_best_time_series_models, evaluate_time_series_models)
from src.preprocessing.download_data import ensure_dataset_exists
from src.preprocessing.preprocess_data import preprocess_data
from src.preprocessing.split_dataset import split_dataset


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    try:
        # Load configuration and setup logging
        config = load_config()
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting the pipeline")

        # Define paths
        dataset_path = Path(config["data"]["local_path"])
        output_path_regression = Path(
            "data/us_pollution_data_preprocessed_regression.csv"
        )
        output_path_time_series = Path(
            "data/us_pollution_data_preprocessed_time_series.csv"
        )
        dataset_url = config["data"]["source_url"]

        # Ensure dataset exists
        ensure_dataset_exists(dataset_path, dataset_url)

        # Preprocess data
        logger.info("Preprocessing data")
        time_series_data, regression_data, encoders = preprocess_data(
            file_path=str(dataset_path),
            output_path_regression=str(output_path_regression),
            output_path_timeseries=str(output_path_time_series),
            input_headers=config["data"]["input_headers"],
            pollutant_headers=config["data"]["pollutant_headers"],
        )

        # Split datasets
        logger.info("Splitting datasets")
        train_data_regression, test_data_regression = split_dataset(
            file_path=str(output_path_regression),
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
        )

        train_data_time_series, test_data_time_series = split_dataset(
            file_path=str(output_path_time_series),
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            date_column="date",
        )

        # Train models
        logger.info("Training models")
        regression_models = train_regression_models(
            train_data=train_data_regression,
            input_headers=config["data"]["input_headers"],
            pollutant_headers=config["data"]["pollutant_headers"],
            model_configs=config["models"].get("regression", {}),
        )

        time_series_models = train_time_series_models(
            train_data=train_data_time_series,
            pollutant_headers=config["data"]["pollutant_headers"],
            model_configs=config["models"].get("time_series", {}),
        )

        # Evaluate models
        logger.info("Evaluating models")
        regression_evaluation_metrics = evaluate_regression_models(
            models=regression_models,
            test_data=test_data_regression,
            input_headers=config["data"]["input_headers"],
            pollutant_headers=config["data"]["pollutant_headers"],
        )

        time_series_evaluation_metrics = evaluate_time_series_models(
            models=time_series_models,
            test_data=test_data_time_series,
            train_data=train_data_time_series,
            pollutant_headers=config["data"]["pollutant_headers"],
        )

        # Select best models
        best_regression_models = choose_best_regression_models(
            metrics=regression_evaluation_metrics
        )
        best_time_series_models = choose_best_time_series_models(
            metrics=time_series_evaluation_metrics
        )

        # Plot predictions
        logger.info("Plotting predictions")
        plot_regression_predictions(
            models=best_regression_models,
            test_data=test_data_regression,
            input_headers=config["data"]["input_headers"],
            pollutant_headers=config["data"]["pollutant_headers"],
            specific_cities=config["data"]["specific_cities"],
            encoders=encoders,
        )

        plot_time_series_predictions(
            models=best_time_series_models,
            test_data=test_data_time_series,
            pollutant_headers=config["data"]["pollutant_headers"],
        )

        # Complete pipeline
        logger.info("Pipeline completed successfully")
        logger.info(f"Regression Evaluation Metrics: {regression_evaluation_metrics}")
        logger.info(f"Time Series Evaluation Metrics: {time_series_evaluation_metrics}")

    except Exception as e:
        logger.exception("An error occurred during the pipeline execution")
        raise


if __name__ == "__main__":
    main()
