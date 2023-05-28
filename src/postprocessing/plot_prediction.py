import pandas as pd
from matplotlib import pyplot as plt


def prepare_and_plot_data(
    historical_start_year,
    future_start_date,
    future_end_date,
    state,
    city,
    best_model,
    dataset,
    state_mapper,
    city_mapper,
):
    historical = dataset.loc[
        (dataset["Year"] >= historical_start_year)
        & (dataset["City"] == city_mapper.transform([city])[0])
        & (dataset["State"] == state_mapper.transform([state])[0])
    ]

    future_dates = pd.period_range(
        start=future_start_date, end=future_end_date, freq="M"
    )
    future = pd.DataFrame()
    future["YearMonth"] = future_dates
    future["Year"] = future_dates.year
    future["Month"] = future_dates.month
    future["State"] = state_mapper.transform([state])[0]
    future["City"] = city_mapper.transform([city])[0]

    future_aqis = best_model.predict(future[["Year", "Month", "State", "City"]])

    for i, header in enumerate(["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"]):
        future[header] = future_aqis[:, i]

    fig, axs = plt.subplots(
        len(["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"]),
        figsize=(15, 5 * len(["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"])),
    )

    for i, header in enumerate(["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"]):
        historical_agg = (
            historical.groupby(["Year", "Month"])[header].mean().reset_index()
        )
        historical_agg["YearMonth"] = (
            historical_agg["Year"] * 100 + historical_agg["Month"]
        )

        axs[i].plot(
            historical_agg["YearMonth"],
            historical_agg[header],
            label=f"Historical {header}",
        )
        axs[i].plot(
            future["Year"] * 100 + future["Month"],
            future[header],
            label=f"Predicted {header}",
        )
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel("AQI")
        axs[i].legend()

    plt.tight_layout()
    plt.show()
