import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.clean_dataset import clean_dataset
from src.preprocessing.split_dataset import split_dataset


def load_and_preprocess_data(path, input_headers, output_headers):
    dataset = pd.read_csv(path)

    dataset["Date Local"] = pd.to_datetime(dataset["Date Local"])
    dataset["Year"] = dataset["Date Local"].dt.year
    dataset["Month"] = dataset["Date Local"].dt.month

    state_mapper = LabelEncoder()
    city_mapper = LabelEncoder()

    dataset["State"] = state_mapper.fit_transform(dataset["State"])
    dataset["City"] = city_mapper.fit_transform(dataset["City"])

    columns_to_keep = input_headers + output_headers
    dataset = clean_dataset(dataset, columns_to_keep)

    x_dataset = dataset[input_headers]
    y_dataset = dataset[output_headers]

    dataset_split = split_dataset(x_dataset, y_dataset)

    return dataset, dataset_split, state_mapper, city_mapper
