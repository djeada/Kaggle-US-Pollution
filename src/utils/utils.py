from typing import Dict, List

import pandas as pd


def filter_data_by_cities(
    data: pd.DataFrame, specific_cities: List[Dict[str, str]]
) -> pd.DataFrame:
    city_filters = [
        f"(state == '{city['state']}') & (city == '{city['city']}')"
        for city in specific_cities
    ]
    query = " | ".join(city_filters)
    filtered_data = data.query(query)
    result = filtered_data.drop_duplicates().reset_index(drop=True)
    return result
