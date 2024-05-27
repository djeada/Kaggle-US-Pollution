import logging
import yaml
from pathlib import Path
from src.preprocessing.download_data import assert_dataset_exists
from src.preprocessing.preprocess_data import preprocess_data
from src.preprocessing.split_dataset import split_dataset
from models.model_training import train_model, evaluate_model
from postprocessing.plot_prediction import plot_predictions
from config.logging_config import setup_logging


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = load_config()
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting the pipeline")

    dataset_path = Path(config["data"]["local_path"])
    output_path = Path("data/us_pollution_data_preprocessed.csv")
    dataset_url = config["data"]["source_url"]

    # Ensure dataset exists
    assert_dataset_exists(dataset_path, dataset_url)

    # Preprocess the data (which includes cleaning)
    logger.info("Preprocessing data")
    cleaned_data, state_mapper, city_mapper = preprocess_data(
        file_path=str(dataset_path),
        output_path=str(output_path),
        input_headers=config["data"]["input_headers"],
        pollutant_headers=config["data"]["pollutant_headers"],
    )

    # Split the data
    logger.info("Splitting data")
    train_data, test_data = split_dataset(
        file_path=str(output_path),
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Train the models
    logger.info("Training models")
    models = train_model(
        train_data=train_data,
        input_headers=config["data"]["input_headers"],
        pollutant_headers=config["data"]["pollutant_headers"],
        config=config["model"],
    )

    # Evaluate the models
    logger.info("Evaluating models")
    evaluation_metrics = evaluate_model(
        models=models,
        test_data=test_data,
        input_headers=config["data"]["input_headers"],
        pollutant_headers=config["data"]["pollutant_headers"],
    )

    # Plot predictions
    logger.info("Plotting predictions")
    plot_predictions(
        models=models,
        test_data=test_data,
        input_headers=config["data"]["input_headers"],
        pollutant_headers=config["data"]["pollutant_headers"],
        specific_cities=config["data"]["specific_cities"],
        state_mapper=state_mapper, city_mapper=city_mapper
    )

    logger.info("Pipeline completed successfully")
    logger.info(f"Evaluation Metrics: {evaluation_metrics}")


if __name__ == "__main__":
    main()
