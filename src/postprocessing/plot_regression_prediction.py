import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def generate_full_date_range(city_data):
    min_year, max_year = city_data["year"].min(), city_data["year"].max()
    full_date_range = pd.DataFrame(
        {
            "year": np.repeat(np.arange(min_year, max_year + 1), 12),
            "month": np.tile(np.arange(1, 13), max_year - min_year + 1),
        }
    )
    return full_date_range


def plot_city_predictions(
    models: Dict[str, Any],
    input_headers: List[str],
    pollutant_headers: List[str],
    city_data: pd.DataFrame,
    state: str,
    city: str,
    numeric_state,
    numeric_city,
    n: int = 3,
) -> None:
    """
    Plots actual vs predicted values for pollutants in a specific city, including future predictions.

    Parameters:
        models (Dict[str, Any]): Dictionary of trained models.
        input_headers (List[str]): List of input features used for training.
        pollutant_headers (List[str]): List of pollutant columns to predict.
        city_data (pd.DataFrame): Data for the specific city.
        state (str): State of the city.
        city (str): City name.
    """
    if not models:
        return

    # Ensure the year and month columns are integers
    city_data = city_data.copy()
    city_data["year"] = city_data["year"].astype(int)
    city_data["month"] = city_data["month"].astype(int)

    # Generate full date range and merge with city_data to fill missing months/years
    full_date_range = generate_full_date_range(city_data)
    city_data = pd.merge(full_date_range, city_data, on=["year", "month"], how="left")
    city_data = city_data.fillna(0)  # Fill NaN values with zeroes

    # Aggregate data by year and month, calculating the mean for each month
    city_data_agg = city_data.groupby(["year", "month"]).mean().reset_index()

    # Sort data by year and month
    city_data_agg = city_data_agg.sort_values(by=["year", "month"])

    # Create a year_month column for plotting
    city_data_agg["year_month"] = city_data_agg.apply(
        lambda row: f"{row['year']}-{int(row['month']):02d}", axis=1
    )

    X_test = city_data_agg[input_headers]

    for pollutant in pollutant_headers:
        y_test = city_data_agg[pollutant]
        model = models[pollutant]
        predictions = model.predict(X_test)

        # Extend predictions 10 years into the future
        last_year = city_data_agg["year"].max()
        future_years = np.arange(last_year + 1, last_year + 11)
        future_months = np.tile(np.arange(1, 13), len(future_years))
        future_years = np.repeat(future_years, 12)
        future_data = pd.DataFrame({"year": future_years, "month": future_months})

        future_data["year"] = future_data["year"].astype(int)
        future_data["month"] = future_data["month"].astype(int)

        # Add state and city to the future_data
        future_data["state"] = numeric_state
        future_data["city"] = numeric_city

        # Ensure future_data contains all input_headers columns
        for header in input_headers:
            if header not in future_data.columns:
                future_data[header] = 0  # Or any default value as needed

        future_X = future_data[input_headers]
        future_predictions = model.predict(future_X)
        future_data["predicted"] = future_predictions

        # Sort future_data by year and month
        future_data = future_data.sort_values(by=["year", "month"])

        # Select every n-th month
        selected_indices = np.arange(0, len(city_data_agg), n)
        future_selected_indices = np.arange(0, len(future_data), n)

        plt.figure(figsize=(14, 10))
        # Bar plot for actual and predicted values
        width = 0.4  # Bar width
        x = np.arange(len(selected_indices))

        plt.bar(
            x - width / 2, y_test.iloc[selected_indices], width=width, label="Actual"
        )
        plt.bar(
            x + width / 2, predictions[selected_indices], width=width, label="Predicted"
        )

        # Add future predictions as a separate bar
        future_x = np.arange(
            len(selected_indices), len(selected_indices) + len(future_selected_indices)
        )
        plt.bar(
            future_x,
            future_data["predicted"].iloc[future_selected_indices],
            width=width,
            label="Future Predicted",
            alpha=0.6,
        )

        # Fit and plot a linear regression line over the data
        X = np.arange(len(selected_indices) + len(future_selected_indices)).reshape(
            -1, 1
        )
        y_combined = np.concatenate(
            [
                y_test.iloc[selected_indices],
                future_data["predicted"].iloc[future_selected_indices],
            ]
        )
        reg = LinearRegression().fit(X, y_combined)
        plt.plot(
            X, reg.predict(X), color="red", linestyle="--", label="Linear Regression"
        )

        plt.xlabel("Time (Year-Month)")
        plt.ylabel(pollutant)
        plt.title(
            f"Actual vs Predicted for {pollutant} in State: {state}, City: {city}"
        )
        plt.legend()

        # Adjusting x-ticks to show one label per year for readability
        years = city_data_agg["year"].unique()
        year_labels = [f"{year}-01" for year in years]
        year_positions = [
            city_data_agg[city_data_agg["year"] == year].index[0] // n for year in years
        ]

        future_years = future_data["year"].unique()
        future_year_labels = [f"{year}-01" for year in future_years]
        future_year_positions = [
            len(selected_indices)
            + future_data[future_data["year"] == year].index[0] // n
            for year in future_years
        ]

        all_year_labels = year_labels + future_year_labels
        all_year_positions = year_positions + future_year_positions

        plt.xticks(ticks=all_year_positions, labels=all_year_labels, rotation=90)

        plt.tight_layout()
        plt.savefig(f"data/predictions_plot_{state}_{city}_{pollutant}.png")
        plt.show()

        logger.info(
            f"Predictions plotted successfully for {pollutant} in State: {state}, City: {city}"
        )


def plot_overall_predictions(
    models: Dict[str, Any],
    test_data: pd.DataFrame,
    input_headers: List[str],
    pollutant_headers: List[str],
) -> None:
    """
    Plots actual vs predicted values for pollutants over the entire dataset.

    Parameters:
        models (Dict[str, Any]): Dictionary of trained models.
        test_data (pd.DataFrame): The test dataset.
        input_headers (List[str]): List of input features used for training.
        pollutant_headers (List[str]): List of pollutant columns to predict.
    """
    for pollutant in pollutant_headers:
        plt.figure(figsize=(10, 6))
        y_test = test_data[pollutant]
        model = models[pollutant]
        X_test = test_data[input_headers]
        predictions = model.predict(X_test)

        plt.plot(y_test.values[:1000], label=f"Actual {pollutant}", alpha=0.5)
        plt.plot(predictions[:1000], label=f"Predicted {pollutant}", linestyle="--")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.title(f"Actual vs Predicted for {pollutant}")
        plt.legend()
        plt.savefig(f"data/predictions_plot_bundled_{pollutant}.png")
        plt.show()

        logger.info(f"Bundled predictions plotted successfully for {pollutant}")


def plot_regression_predictions(
    models: Dict[str, Any],
    test_data: pd.DataFrame,
    input_headers: List[str],
    pollutant_headers: List[str],
    specific_cities: List[Dict[str, str]],
    encoders,
) -> None:
    """
    Plots the actual vs predicted values for each pollutant for specific cities and overall bundled data.

    Parameters:
        models (Dict[str, Any]): Dictionary of trained models.
        test_data (pd.DataFrame): The test dataset.
        input_headers (List[str]): List of input features used for training.
        pollutant_headers (List[str]): List of pollutant columns to predict.
        specific_cities (List[Dict[str, str]]): List of specific cities to plot.
    """
    try:
        # Map the specified cities to their numeric values
        city_mappings = []
        for city_info in specific_cities:
            state = city_info["state"]
            city = city_info["city"]
            numeric_state = encoders['state'].transform([state])[0]
            numeric_city = encoders['city'].transform([city])[0]
            city_mappings.append((numeric_state, numeric_city))

        # Plot for specific cities
        for numeric_state, numeric_city in city_mappings:
            city_data = test_data[
                (test_data["state"] == numeric_state)
                & (test_data["city"] == numeric_city)
            ]
            if city_data.empty:
                logger.warning(f"No data found for State: {state}, City: {city}")
                continue

            plot_city_predictions(
                models=models,
                input_headers=input_headers,
                pollutant_headers=pollutant_headers,
                city_data=city_data,
                state=state,
                city=city,
                numeric_state=numeric_state,
                numeric_city=numeric_city,
            )

        plot_overall_predictions(
            models=models,
            test_data=test_data,
            input_headers=input_headers,
            pollutant_headers=pollutant_headers,
        )

    except Exception as e:
        logger.error(f"Error plotting predictions: {e}")
        raise
