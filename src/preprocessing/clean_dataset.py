import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_dataset(
    data_frame: pd.DataFrame, state_mapper: LabelEncoder, city_mapper: LabelEncoder
) -> pd.DataFrame:
    """
    Cleans the dataset.

    :param data_frame: The dataset to clean.
    :return: The cleaned dataset.
    """
    data_frame = data_frame.drop(
        [
            "Unnamed: 0",
            "State Code",
            "County Code",
            "Address",
            "Site Num",
            "NO2 Units",
            "O3 Units",
            "SO2 Units",
            "CO Units",
        ],
        axis=1,
        errors="ignore",
    )
    data_frame = data_frame[data_frame.State != "Country Of Mexico"]

    data_frame["year"] = pd.DatetimeIndex(data_frame["Date Local"]).year
    data_frame["month"] = pd.DatetimeIndex(data_frame["Date Local"]).month
    data_frame["Date Local"] = pd.to_datetime(
        data_frame["Date Local"], format="%Y-%m-%d"
    )

    ## use the mappers to encode the states and cities
    data_frame["State"] = data_frame["State"].apply(
        lambda state: state_mapper.transform([state])[0]
    )
    data_frame["City"] = data_frame["City"].apply(
        lambda city: city_mapper.transform([city])[0]
    )

    ## convert all features to numeric
    for column_name in data_frame.columns:
        if data_frame[column_name].dtype == object:
            data_frame[column_name] = pd.to_numeric(
                data_frame[column_name], errors="coerce"
            )

    # fill nan with mean
    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_frame = data_frame.fillna(
        data_frame.mean() if data_frame.mean().notnull().all() else 0
    )

    return data_frame
