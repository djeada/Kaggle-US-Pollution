"""

• Which seasons causes a spike in pollution? Why? Does it vary across Geographical regions in the US?

• Which state and which city has the highest pollution / heighest per Capita?

• Where the increase in pollution is the most significant?

• Is presence of one pollutant correlated with the presence of another pollutant?

Models, predictions and scores
"""

# load the dataset
csv_path = "pollution_us_2000_2016.csv"
df = pd.read_csv(csv_path)
df.head()

# clean data
df = df.drop(
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
df = df[df.State != "Country Of Mexico"]
df.head()

# add some columns, to make further analysis easier
df["year"] = pd.DatetimeIndex(df["Date Local"]).year
df["month"] = pd.DatetimeIndex(df["Date Local"]).month
df["Date Local"] = pd.to_datetime(df["Date Local"], format="%Y-%m-%d")

# investigate how pollution of each of the 4 pollutants varies by state


def plot_mean_by_state(data_frame: pd.DataFrame, pollutant: str, ax: plt.Axes) -> None:
    """
    Plots the mean of a column by state using a bar chart.

    :param data_frame: the data frame holding the data
    :param pollutant: the column to plot
    :param ax: the axis to plot on
    """

    states = data_frame["State"].unique()

    state_mean = {}
    for state in states:
        state_mean[state] = data_frame[data_frame["State"] == state][col_name].mean()

    state_mean = sorted(state_mean.items(), key=lambda x: x[1], reverse=True)

    states = [x[0] for x in state_mean]
    means = [x[1] for x in state_mean]
    ax.bar(states, means, color=plt.cm.jet(np.linspace(1, 0, len(state_mean))))
    ax.tick_params(labelrotation=90)
    ax.set_xlabel("State")
    ax.set_ylabel(f"{col_name}")


# plot the mean of each pollutant by state on one figure
column_names = ["NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean"]
fig, axs = plt.subplots(4, 1, figsize=(10, 15))
fig.suptitle("Pollution by State")

for i, col_name in enumerate(column_names):
    plot_mean_by_state(df, col_name, axs[i])

plt.subplots_adjust(hspace=1.5)
plt.show()

# 5 most polluted states by each pollutant
def get_top_5_states(data_frame: pd.DataFrame, pollutant: str) -> List[str]:
    """
    Returns the top 5 states with the highest mean of a given pollutant.

    :param data_frame: the data frame holding the data
    :param pollutant: the column to plot
    :return: a list of the top 5 states
    """

    states = data_frame["State"].unique()

    state_mean = {}
    for state in states:
        state_mean[state] = data_frame[data_frame["State"] == state][col_name].mean()

    state_mean = sorted(state_mean.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in state_mean[:5]]


column_names = ["NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean"]
for col_name in column_names:
    print(f"Top 5 states for {col_name}: {get_top_5_states(df, col_name)}")

# 5 most polluted cities by each pollutant
def get_top_5_cities(data_frame: pd.DataFrame, pollutant: str) -> List[str]:
    """
    Returns the top 5 cities with the highest mean of a given pollutant.

    :param data_frame: the data frame holding the data
    :param pollutant: the column to plot
    :return: a list of the top 5 cities
    """

    cities = data_frame["City"].unique()

    city_mean = {}
    for city in cities:
        city_mean[city] = data_frame[data_frame["City"] == city][col_name].mean()

    city_mean = sorted(city_mean.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in city_mean[:5]]


column_names = ["NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean"]
for col_name in column_names:
    print(f"Top 5 cities for {col_name}: {get_top_5_cities(df, col_name)}")

# let's see how correlated the pollutants are
def plot_correlation_matrix(data_frame: pd.DataFrame, collumn_names: List[str]) -> None:
    """
    Plots a correlation matrix for a given list of columns.

    :param data_frame: the data frame holding the data
    :param collumn_names: the columns to plot
    """
    temp_df = data_frame[collumn_names]
    temp_df = data_frame.dropna()
    corr = temp_df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    plt.show()


plot_correlation_matrix(df, ["NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean"])


def sort_by_correlation(df, n=-1):
    """
    Sorts the columns of a data frame by correlation with the target column.

    :param df: the data frame holding the data
    :param n: the number of columns to return
    :return: a list of the columns sorted by correlation
    """

    def get_redundant_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Get diagonal and lower triangular pairs of correlation matrix.

        :param df: the data frame holding the data
        :return: a list of tuples of the redundant pairs
        """
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[:n]


print("Top Absolute Correlations")
top_correlations = sort_by_correlation(
    df[["NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean"]]
)
print(top_correlations)

# Plot the most correlated pollutants
# and fit a linear regression model to the data
most_correlated_pair = top_correlations.index[0][0], top_correlations.index[0][1]
print(f"Most correlated pair: {most_correlated_pair}")

pollutant_1, pollutant_2 = most_correlated_pair

X = df_c[pollutant_1].values.reshape(-1, 1)
y = df_c[pollutant_2].values.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X, y)

domain = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
y_pred = reg.predict(domain)

plt.figure(figsize=(10, 10))
plt.scatter(X, y)
plt.plot(domain, y_pred, linewidth=3, color="orange")
plt.xlabel(pollutant_1)
plt.ylabel(pollutant_2)
plt.title(f"{pollutant_1} vs {pollutant_2}")
# display the line coefficients in form of y = mx + b
plt.legend(["data", f"y = {reg.coef_[0][0]:.2f}x + {reg.intercept_[0]:.2f}"])
plt.show()

# Let's see if the air quality varies by season
## Plot 4 AQIs with top 4 states


def plot_aqi_by_state(data_frame: pd.DataFrame, pollutant: str) -> None:
    """
    Plots the AQI of a given pollutant for the top 4 states.

    :param data_frame: the data frame holding the data
    :param pollutant: the column to plot
    """
    df_c = data_frame[["State", f"{pollutant} AQI"]]
    df_c = df_c.dropna()
    df_c = df_c.groupby("State").mean()
    df_c = df_c.sort_values(f"{pollutant} AQI", ascending=False)
    df_c = df_c.head(4)
    for state in df_c.index:
        df_s = data_frame[data_frame["State"] == state]
        df_s = df_s[["Date Local", f"{pollutant} AQI"]]
        df_s = df_s.dropna()
        df_s = df_s.groupby("Date Local").mean()
        # get average value for every month
        df_s = df_s.resample("M").mean()
        plt.plot(df_s.index, df_s[f"{pollutant} AQI"], label=state)
    plt.xlabel("Date")
    plt.ylabel(f"{pollutant} AQI")
    plt.title(f"{pollutant} AQI")
    plt.legend()


fig = plt.figure(figsize=(12, 8))
fig.tight_layout()

plot_positions = (221, 222, 223, 224)
pollutants = ["NO2", "O3", "SO2", "CO"]
for plot_position, pollutant in zip(plot_positions, pollutants):
    plt.subplot(plot_position)
    plot_aqi_by_state(df, pollutant)

plt.subplots_adjust(hspace=0.3)
plt.show()

# What season is the worst for air quality?

seasons = {
    "Winter": (1, 2, 12),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Autumn": (9, 10, 11),
}


def get_season(month: int) -> str:
    """
    Returns the season for a given month.

    :param month: the month to get the season for
    :return: the season
    """
    for season, months in seasons.items():
        if month in months:
            return season


# for each year check in which season the air quality is the worst
# and print the results

pollutants = ["NO2", "O3", "SO2", "CO"]
for pollutant in pollutants:
    print(f"Pollutant: {pollutant}")

    histogram = {}

    for year in range(2010, 2016):
        df_y = df[df["Date Local"].dt.year == year]
        df_y = df_y.dropna()
        df_y = df_y.groupby("Date Local").mean(numeric_only=True)
        df_y = df_y.resample("M").mean(numeric_only=True)
        df_y["Season"] = [get_season(date.month) for date in df_y.index]
        df_y = df_y.groupby("Season").mean(numeric_only=True)
        df_y = df_y.sort_values(f"{pollutant} AQI", ascending=False)
        season = df_y.index[0]
        if season in histogram:
            histogram[season] += 1
        else:
            histogram[season] = 1

    # find the season with the most occurrences
    print(f"Worst season: {max(histogram, key=histogram.get)}")
    print()

# Is the air quality getting better or worse over time?
# fit a linear regression model to the data
# and check if the slope is positive or negative
# use the whole data without grouping by state or city

pollutants = ["NO2", "O3", "SO2", "CO"]
for pollutant in pollutants:
    print(f"Pollutant: {pollutant}")

    temp_df = df[[f"{pollutant} AQI", "Date Local"]]
    temp_df = temp_df.dropna()

    X = temp_df[f"{pollutant} AQI"].values.reshape(-1, 1)
    y = temp_df["Date Local"].values.reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(X, y)

    print(
        f'Slope of the fitted regression line: {reg.coef_[0][0]} is {"positive" if reg.coef_[0][0] > 0 else "negative"}.'
    )
    print()


def create_map():
    # create the map
    us_map = Basemap(
        llcrnrlon=-119,
        llcrnrlat=22,
        urcrnrlon=-64,
        urcrnrlat=49,
        projection="lcc",
        lat_1=33,
        lat_2=45,
        lon_0=-95,
    )

    # load the shapefile, use the name 'states'
    us_map.readshapefile("st99_d00", name="states", drawbounds=True)

    return us_map


def update_state_color(us_map, state_index: int, color: str, ax):
    seg = us_map.states[state_index]
    poly = Polygon(seg, facecolor=color, edgecolor=color)
    ax.add_patch(poly)


def plot_aqi_per_state(data_frame, column_name, us_map, ax):
    # collect the state names from the shapefile attributes so we can
    # look up the shape obect for a state by it's name
    state_names = []
    for shape_dict in us_map.states_info:
        state_names.append(shape_dict["NAME"])

    # now we want to plot the SO2 AQI for each state using gradient color
    # first we need to get the average AQI for each state at specific date
    df_c = data_frame[["State", column_name]]
    df_c = df_c.dropna()
    df_c = df_c.groupby("State").mean()
    df_c = df_c.sort_values(column_name, ascending=False)

    # now we need to get the min and max AQI values
    min_aqi = df_c[column_name].min()
    max_aqi = df_c[column_name].max()

    # now we need to get the color for each state
    for state in df_c.index:
        if state not in state_names:
            continue
        # get the AQI value for the state
        aqi = df_c.loc[state, column_name]
        # get the color for the AQI value
        def get_color(aqi: float, min_aqi: float, max_aqi: float):
            # get the normalized value
            norm = (aqi - min_aqi) / (max_aqi - min_aqi)
            # get the color
            color = plt.cm.coolwarm(norm)
            return color

        color = get_color(aqi, min_aqi, max_aqi)
        # update the color for the state
        update_state_color(us_map, state_names.index(state), color, ax)


## create 4 plots
chosen_date = "2015-06-18"
# drop all rows that don't have the chosen date
temp_df = df[df["Date Local"] == chosen_date]

fig = plt.figure(figsize=(12, 8))
fig.suptitle(f"Air quality on {chosen_date}", fontsize=16)

plot_positions = (221, 222, 223, 224)
plot_titles = ("SO2 AQI", "NO2 AQI", "O3 AQI", "CO AQI")

for plot_position, plot_title in zip(plot_positions, plot_titles):
    # create the plot
    ax = fig.add_subplot(plot_position)
    ax.set_title(plot_title)
    # create the map
    us_map = create_map()
    # plot the AQI per state
    plot_aqi_per_state(temp_df, plot_title, us_map, ax)


plt.show()
