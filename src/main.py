from pathlib import Path
import pandas as pd

from src.models.lasso_regression import Lasso
from src.models import LinearRegression
from src.models.random_forest_regression import RandomForest
from src.models.xgb_regression import XGB
from src.postprocessing.performance_metrics import evaluate_models
from src.postprocessing.plot_prediction import prepare_and_plot_data
from src.preprocessing.preprocess_data import load_and_preprocess_data

DATASET_PATH = Path("../resources/pollution_us_2000_2016.csv")
INPUT_HEADERS = ["Year", "Month", "State", "City"]
OUTPUT_HEADERS = ["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"]


def train_models(dataset_split):
    model_types = [LinearRegression, Lasso, RandomForest, XGB]
    models = []

    for model_type in model_types:
        print(f"Training {model_type.__name__} model...")
        model = model_type()
        model.fit(dataset_split.train_x.to_numpy(), dataset_split.train_y.to_numpy())
        models.append(model)
        print(f"{model_type.__name__} model trained successfully.")

    return models


def make_prediction(best_model, year, month, state, city, state_mapper, city_mapper):
    prediction = best_model.predict(
        [
            [
                year,
                month,
                state_mapper.transform([state])[0],
                city_mapper.transform([city])[0],
            ]
        ]
    )
    return prediction


def main():
    print("Loading and preprocessing the data...")
    dataset, dataset_split, state_mapper, city_mapper = load_and_preprocess_data(
        DATASET_PATH, INPUT_HEADERS, OUTPUT_HEADERS
    )

    print("Training the models...")
    models = train_models(dataset_split)

    print("Evaluating the models...")
    best_model, scores = evaluate_models(models, dataset_split)
    print(
        f"The best model is {best_model['model']} with an r2_score of {best_model['r2_score']}"
    )

    print("Making a prediction for a specific date, city, and state...")
    year, month, state, city = (
        pd.to_datetime("2040-01-01").year,
        pd.to_datetime("2040-01-01").month,
        "California",
        "Los Angeles",
    )
    prediction = make_prediction(
        best_model["model_instance"],
        year,
        month,
        state,
        city,
        state_mapper,
        city_mapper,
    )
    print(f"The predicted AQI for {city}, {state} on 2040-01-01 is {prediction[0]}")

    print(
        "Preparing and plotting historical and future data for the selected city and state..."
    )
    prepare_and_plot_data(
        2000,
        "2020-01-01",
        "2040-12-31",
        state,
        city,
        best_model["model_instance"],
        dataset,
        state_mapper,
        city_mapper,
    )
    print("Plotting complete.")


if __name__ == "__main__":
    main()
