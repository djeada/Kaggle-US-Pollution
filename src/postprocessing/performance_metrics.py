# performance_metrics.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, test_data):
    try:
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        logger.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")
        return {"mse": mse, "r2": r2}
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise
