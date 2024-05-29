import logging
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from src.utils.utils import filter_data_by_cities

logger = logging.getLogger(__name__)


def plot_time_series_predictions(models, test_data, pollutant_headers):
    for pollutant, info in models.items():
        model_type, model_info, model = (
            info["model_type"],
            info["metadata"],
            info["model"],
        )
        state, city = model_info["state"], model_info["city"]
        specific_cities = [{"state": state, "city": city}]
        city_test_data = filter_data_by_cities(test_data, specific_cities)

        if city_test_data.empty:
            logger.warning(f"No test data found for city: {city}, state: {state}")
            continue

        city_test_data["date"] = pd.to_datetime(city_test_data["date"])
        city_test_data.set_index("date", inplace=True)

        y_test = city_test_data[pollutant]

        if isinstance(model, (ARIMAResultsWrapper, SARIMAXResultsWrapper)):
            predictions = model.get_forecast(steps=len(y_test)).predicted_mean
        else:
            raise ValueError(f"Model type {model_type} not supported for time series")

        # Align indices of y_test and predictions
        y_test, predictions = y_test.align(predictions, join="inner")

        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test, label="Actual", color="blue")
        plt.plot(predictions.index, predictions, label="Predicted", color="red")
        plt.title(f"Best Model Predictions for {pollutant} in {city}, {state}")
        plt.xlabel("Date")
        plt.ylabel(f"{pollutant} Levels")
        plt.legend()
        plt.grid(True)
        plt.show()
