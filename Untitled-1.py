

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

path = Path("resources/pollution_us_2000_2016.csv")

# check if file exists and if not download it https://www.kaggle.com/datasets/sogun3/uspollution/download?datasetVersionNumber=1
if not path.exists():
    print("Downloading file...")
    url = "https://www.kaggle.com/datasets/sogun3/uspollution/download?datasetVersionNumber=1"
    r = requests.get(url, allow_redirects=True)
    zip_path = path.parent / "pollution_us_2000_2016.zip"
    open(zip_path, 'wb').write(r.content)
    print("Download complete.")

    # unzip file
    with zip_path.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall("resources")
    
    # remove zip file
    zip_path.unlink()
    


csv_path = "pollution_us_2000_2016.csv"
df = pd.read_csv(csv_path)
df.head()

# Delete extraneous column
df = df.drop(['Unnamed: 0','State Code','County Code','Address','Site Num','NO2 Units','O3 Units','SO2 Units','CO Units'], axis=1, errors='ignore')
df = df[df.State!='Country Of Mexico']
df.head()

# split the data
df['year'] = pd.DatetimeIndex(df['Date Local']).year
df['month'] = pd.DatetimeIndex(df['Date Local']).month
df['Date Local'] = pd.to_datetime(df['Date Local'],format='%Y-%m-%d')




def mean_values_per_state(data_frame, col_name):
    states = data_frame['State'].unique()

    state_mean_dict = {}
    for state in states:
        state_mean_dict[state] = data_frame[data_frame['State'] == state][col_name].mean()

    state_mean_pairs = sorted(state_mean_dict.items(), key=lambda x: x[1], reverse=True)
    return state_mean_pairs


def plot_mean_by_state(col_name, state_mean_pairs):
    plt.figure(figsize=(20,10))

    states = [state for state, _ in state_mean_pairs]
    means = [mean for _, mean in state_mean_pairs]

    plt.bar(states, means, color=plt.cm.jet(np.linspace(1, 0, len(state_mean_NO2))))
    plt.xticks(rotation=90)
    plt.xlabel('State')
    plt.ylabel(f'{col_name}')
    plt.title(f'{col_name} by State')
    plt.show()

#NO2 data
NO2_data = df[["State","Date Local","NO2 Mean","NO2 1st Max Value", "NO2 1st Max Hour", "NO2 AQI","year"
                          ]]
df_c = NO2_data.sort_values('State')
df_c = NO2_data.sort_values('NO2 Mean')

state_mean_NO2 = mean_values_per_state(df_c, 'NO2 Mean')
plot_mean_by_state('NO2 Mean', state_mean_NO2)

#O3 data
O3_data = df[["State","Date Local","O3 Mean","O3 1st Max Value", "O3 1st Max Hour", "O3 AQI","year"
                            ]]
df_c = O3_data.sort_values('State')
df_c = O3_data.sort_values('O3 Mean')

state_mean_O3 = mean_values_per_state(df_c, 'O3 Mean')
plot_mean_by_state('O3 Mean', state_mean_O3)

#SO2 data
SO2_data = df[["State","Date Local","SO2 Mean","SO2 1st Max Value", "SO2 1st Max Hour", "SO2 AQI","year"
                            ]]
df_c = SO2_data.sort_values('State')
df_c = SO2_data.sort_values('SO2 Mean')

state_mean_SO2 = mean_values_per_state(df_c, 'SO2 Mean')
plot_mean_by_state('SO2 Mean', state_mean_SO2)

#CO data
CO_data = df[["State","Date Local","CO Mean","CO 1st Max Value", "CO 1st Max Hour", "CO AQI","year"
                            ]]
df_c = CO_data.sort_values('State')
df_c = CO_data.sort_values('CO Mean')

state_mean_CO = mean_values_per_state(df_c, 'CO Mean')
plot_mean_by_state('CO Mean', state_mean_CO)

## find and plot correlation between NO2 and SO2

# find the correlation between NO2 and SO2
df_c = df[["NO2 Mean","SO2 Mean"]]
df_c = df_c.dropna()
df_c.corr()

# plot the correlation
plt.figure(figsize=(10,10))
plt.scatter(df_c['NO2 Mean'], df_c['SO2 Mean'])
plt.xlabel('NO2 Mean')
plt.ylabel('SO2 Mean')

## find and plot correlation between O3 and CO

# find the correlation between O3 and CO
df_c = df[["O3 Mean","CO Mean"]]
df_c = df_c.dropna()    
df_c.corr()

# plot the correlation
plt.figure(figsize=(10,10))
plt.scatter(df_c['O3 Mean'], df_c['CO Mean'])
plt.xlabel('O3 Mean')
plt.ylabel('CO Mean')

# find the correlation between all pairs of [NO2, O3, SO2, CO] and find the 2 most correlated pairs
df_c = df[["NO2 Mean","O3 Mean","SO2 Mean","CO Mean"]]
df_c = df_c.dropna()
df_c.corr()

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df_c, 3))

# NO2 vs CO is the most correlated pair
# plot the correlation

# fit linear regression model and plot the line
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df_c['NO2 Mean'].values.reshape(-1,1)
y = df_c['CO Mean'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)

domain = np.linspace(np.min(X), np.max(X), 1000).reshape(-1,1)
y_pred = reg.predict(domain)

plt.figure(figsize=(10,10)) 
plt.scatter(X, y)
plt.plot(domain, y_pred, linewidth=3, color='orange')
plt.xlabel('NO2 Mean')
plt.ylabel('CO Mean')
plt.title('NO2 vs CO')
# display the line coefficients in form of y = mx + b
plt.legend(['data',
    f'y = {reg.coef_[0][0]:.2f}x + {reg.intercept_[0]:.2f}'])
plt.show()

## Plot 4 AQIs with top 4 states
plt.figure(figsize=(12,8))

# NO2 AQI
plt.subplot(2,2,1)
df_c = df[["State","NO2 AQI"]]
df_c = df_c.dropna()
df_c = df_c.groupby('State').mean()
df_c = df_c.sort_values('NO2 AQI', ascending=False)
df_c = df_c.head(4)
# now for the top 4 state plot all the AQI values from 2000 to 2016
for state in df_c.index:
    df_s = df[df['State'] == state]
    df_s = df_s[["Date Local","NO2 AQI"]]
    df_s = df_s.dropna()
    df_s = df_s.groupby('Date Local').mean()
    # get average value for every month
    df_s = df_s.resample('M').mean()
    plt.plot(df_s.index, df_s['NO2 AQI'], label=state)
plt.xlabel('Date')
plt.ylabel('NO2 AQI')
plt.title('NO2 AQI')
plt.legend()

# O3 AQI
plt.subplot(2,2,2)
df_c = df[["State","O3 AQI"]]
df_c = df_c.dropna()
df_c = df_c.groupby('State').mean()
df_c = df_c.sort_values('O3 AQI', ascending=False)  
df_c = df_c.head(4)

for state in df_c.index:
    df_s = df[df['State'] == state]
    df_s = df_s[["Date Local","O3 AQI"]]
    df_s = df_s.dropna()
    df_s = df_s.groupby('Date Local').mean()
    # get average value for every month
    df_s = df_s.resample('M').mean()
    plt.plot(df_s.index, df_s['O3 AQI'], label=state)
plt.xlabel('Date')
plt.ylabel('O3 AQI')
plt.title('O3 AQI')
plt.legend()

# SO2 AQI
plt.subplot(2,2,3)
df_c = df[["State","SO2 AQI"]]
df_c = df_c.dropna()
df_c = df_c.groupby('State').mean()
df_c = df_c.sort_values('SO2 AQI', ascending=False)
df_c = df_c.head(4)

for state in df_c.index:
    df_s = df[df['State'] == state]
    df_s = df_s[["Date Local","SO2 AQI"]]
    df_s = df_s.dropna()
    df_s = df_s.groupby('Date Local').mean()
    # get average value for every month
    df_s = df_s.resample('M').mean()
    plt.plot(df_s.index, df_s['SO2 AQI'], label=state)
plt.xlabel('Date')
plt.ylabel('SO2 AQI')
plt.title('SO2 AQI')
plt.legend()

# CO AQI
plt.subplot(2,2,4)
df_c = df[["State","CO AQI"]]
df_c = df_c.dropna()
df_c = df_c.groupby('State').mean()
df_c = df_c.sort_values('CO AQI', ascending=False)
df_c = df_c.head(4)

for state in df_c.index:
    df_s = df[df['State'] == state]
    df_s = df_s[["Date Local","CO AQI"]]
    df_s = df_s.dropna()
    df_s = df_s.groupby('Date Local').mean()
    # get average value for every month
    df_s = df_s.resample('M').mean()
    plt.plot(df_s.index, df_s['CO AQI'], label=state)
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.title('CO AQI')
plt.legend()

plt.show()



def create_map():
    # create the map
    us_map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    # load the shapefile, use the name 'states'
    us_map.readshapefile('st99_d00', name='states', drawbounds=True)

    return us_map

def update_state_color(us_map, state_index: int, color: str, ax):
    seg = us_map.states[state_index]
    poly = Polygon(seg, facecolor=color, edgecolor=color)
    ax.add_patch(poly)

def plot_aqi_per_state(data_frame, column_name, us_map, ax):
    # collect the state names from the shapefile attributes so we can
    # look up the shape obect for a state by it's name
    state_names = []
    for shape_dict in map.states_info:
        state_names.append(shape_dict['NAME'])

    # now we want to plot the SO2 AQI for each state using gradient color
    # first we need to get the average AQI for each state at specific date of december 2016
    df_c = df[["State","SO2 AQI"]]
    df_c = df_c.dropna()
    df_c = df_c.groupby('State').mean()
    df_c = df_c.sort_values('SO2 AQI', ascending=False)

    # now we need to get the min and max AQI values
    min_aqi = df_c['SO2 AQI'].min()
    max_aqi = df_c['SO2 AQI'].max()

    # now we need to get the color for each state
    for state in df_c.index:
        if state not in state_names:
            continue
        # get the AQI value for the state
        aqi = df_c.loc[state, 'SO2 AQI']
        # get the color for the AQI value
        def get_color(aqi: float, min_aqi: float, max_aqi: float):
            # get the normalized value
            norm = (aqi - min_aqi) / (max_aqi - min_aqi)
            # get the color
            color = plt.cm.coolwarm(1-norm)
            return color
        color = get_color(aqi, min_aqi, max_aqi)
        # update the color for the state
        update_state_color(us_map, state_names.index(state), color, ax)


## create 4 plots
fig = plt.figure()

plot_positions = (221, 222, 223, 224)
plot_titles = ('SO2 AQI', 'NO2 AQI', 'O3 AQI', 'CO AQI')

for plot_position, plot_title in zip(plot_positions, plot_titles):
    # create the plot
    ax = fig.add_subplot(plot_position)
    ax.set_title(plot_title)
    # create the map
    us_map = create_map()
    # plot the AQI per state
    plot_aqi_per_state(df, plot_title, us_map, ax)

    
plt.show()
