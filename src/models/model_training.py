import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def train_single_model(pollutant, X_train, y_train, config):
    """
    Train a single model for a given pollutant.

    :param pollutant: The pollutant to train the model for.
    :param X_train: The training input data.
    :param y_train: The training target data.
    :param config: Configuration for the model.
    :return: Trained model.
    """
    logger.info(
        f"Training model for {pollutant} with hyperparameters: {config['hyperparameters']}"
    )

    model = RandomForestRegressor(**config["hyperparameters"])
    model.fit(X_train, y_train)

    logger.info(f"Model for {pollutant} trained successfully")
    return pollutant, model


def train_model(train_data, input_headers, pollutant_headers, config):
    """
    Train separate models for each pollutant using multiprocessing.

    :param train_data: The training data.
    :param input_headers: List of input features.
    :param pollutant_headers: List of pollutant columns to predict.
    :param config: Configuration for the model.
    :return: Dictionary of trained models.
    """
    models = {}
    pool = Pool()

    tasks = [
        (pollutant, train_data[input_headers], train_data[pollutant], config)
        for pollutant in pollutant_headers
    ]

    results = pool.starmap(train_single_model, tasks)

    for pollutant, model in results:
        models[pollutant] = model

    pool.close()
    pool.join()

    return models


def evaluate_model(models, test_data, input_headers, pollutant_headers):
    """
    Evaluate the trained models.

    :param models: Dictionary of trained models.
    :param test_data: The test data.
    :param input_headers: List of input features.
    :param pollutant_headers: List of pollutant columns to predict.
    :return: Dictionary of evaluation metrics for each model.
    """
    metrics = {}

    for pollutant in pollutant_headers:
        logger.info(f"Evaluating model for {pollutant}")

        X_test = test_data[input_headers]
        y_test = test_data[pollutant]

        model = models[pollutant]
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        metrics[pollutant] = {"mse": mse}
        logger.info(f"Evaluation metrics for {pollutant}: MSE = {mse}")

    return metrics
