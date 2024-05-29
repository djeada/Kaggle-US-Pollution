import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_regression_models(models, test_data, input_headers, pollutant_headers):
    metrics = {pollutant: [] for pollutant in pollutant_headers}

    for pollutant in pollutant_headers:
        for model_type, model in models[pollutant]:
            logger.info(f"Evaluating regression model for {pollutant} ({model_type})")

            X_test = test_data[input_headers]
            y_test = test_data[pollutant]

            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            metrics[pollutant].append((model_type, {"mse": mse, "mae": mae, "r2": r2}))
            logger.info(
                f"Evaluation metrics for {pollutant} ({model_type}): MSE = {mse}, MAE = {mae}, R2 = {r2}"
            )

    return metrics


def choose_best_regression_models(metrics, metric="r2"):

    if not metrics or all(not v for v in metrics.values()):
        return list()

    best_models = {}

    for pollutant, model_metrics in metrics.items():
        best_model = min(
            model_metrics,
            key=lambda x: x[1][metric] if metric != "r2" else -x[1][metric],
        )
        best_models[pollutant] = best_model
        logger.info(
            f"Best regression model for {pollutant}: {best_model[0]} with {metric.upper()} = {best_model[1][metric]}, "
            f"MAE = {best_model[1]['mae']}, R2 = {best_model[1]['r2']}"
        )

    return best_models
