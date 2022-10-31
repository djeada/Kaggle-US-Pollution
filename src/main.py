from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models.lasso import Lasso
from src.models.linear_regression import LinearRegression
from src.postprocessing.performance_metrics import (
    calculate_r2_score,
    calculate_rmse,
    calculate_nrmse,
)
from src.preprocessing.clean_dataset import clean_dataset
from src.preprocessing.download_data import assert_dataset_exists
from src.preprocessing.split_dataset import split_dataset

DATASET_PATH = Path("../resources/pollution_us_2000_2016.csv")
INPUT_HEADERS = ["year", "month", "State", "City"]
OUTPUT_HEADERS = ["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"]


def main():
    print("Preprocessing...")
    assert_dataset_exists(DATASET_PATH)
    output_dir = Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Cleaning dataset...")
    dataset = pd.read_csv(DATASET_PATH)
    state_mapper = LabelEncoder()
    state_mapper.fit_transform(dataset["State"].unique())
    city_mapper = LabelEncoder()
    city_mapper.fit_transform(dataset["City"].unique())
    dataset = clean_dataset(dataset, state_mapper, city_mapper)
    print("Dataset cleaned.")

    # Split the dataset into train and test sets

    print("Splitting dataset...")
    x_dataset = dataset[INPUT_HEADERS]
    y_dataset = dataset[OUTPUT_HEADERS]

    dataset_split = split_dataset(x_dataset, y_dataset, save_to_file=True)

    print("Preprocessing finished.")

    print("Training...")

    model_types = [
        LinearRegression,
        Lasso,
    ]

    models = []

    for model_type in model_types:
        print(f"Training {model_type.__name__}...")
        model = model_type()
        model.fit(dataset_split.train_x.to_numpy(), dataset_split.train_y.to_numpy())
        model.save(f"../output/{model.__class__.__name__}.joblib")
        models.append(model)
    print("Training finished.")

    ### Postprocessing (model -> postprocessing -> metrics)
    print("Postprocessing...")
    scores = []
    for model in models:
        model_name = model.__class__.__name__
        test_y_array = dataset_split.test_y.to_numpy()
        predicted_y_array = model.predict(dataset_split.test_x)
        predicted_y = pd.DataFrame(
            predicted_y_array,
            columns=[f"{header}_predicted" for header in OUTPUT_HEADERS],
        )
        score = {
            "model": model_name,
            "r2_score": calculate_r2_score(test_y_array, predicted_y_array),
            "rmse": calculate_rmse(test_y_array, predicted_y_array),
            "nrmse": calculate_nrmse(test_y_array, predicted_y_array),
        }
        scores.append(score)
        print(score)

    # Find the model with best r2_score
    best_model = max(scores, key=lambda score: score["r2_score"])
    print(f"Best model: {best_model['model']}")
    print("Postprocessing finished.")

    date = pd.to_datetime("2050-01-01")
    year = date.year
    month = date.month
    state = "California"
    city = "Los Angeles"

    # predict the AQI for the given date, state and city
    best_model = models[0]
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
    print(f"On {date} in {city}, {state} the AQI will be: ")
    for i, header in enumerate(OUTPUT_HEADERS):
        print(f"{header}: {prediction[0][i]}")


if __name__ == "__main__":
    main()
