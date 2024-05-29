# Kaggle U.S. Pollution
An analysis of the US Pollution dataset to explore trends, predict future pollution levels, and identify factors influencing air quality in various regions of the United States. This project employs advanced data engineering and machine learning techniques.

![dataset-cover](https://user-images.githubusercontent.com/37275728/198901751-79133990-37e2-4832-a698-3697dce69f50.jpeg)

## Introduction

> This dataset deals with pollution in the U.S. Pollution in the U.S. has been well documented by the U.S. EPA but it is a pain to download all the data and arrange them in a format that interests data scientists. Hence I gathered four major pollutants (Nitrogen Dioxide, Sulphur Dioxide, Carbon Monoxide and Ozone) for every day from 2000 - 2016 and place them neatly in a CSV file. 

<a href="https://www.kaggle.com/datasets/sogun3/uspollution">Read more.</a>

## Installation

Follow these steps to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/djeada/kaggle-us-pollution.git
    ```

2. Navigate to the project directory:
    ```bash
    cd kaggle-us-pollution
    ```

3. Install `virtualenv` if it's not already installed:
    ```bash
    pip install virtualenv
    ```

4. Create and activate a virtual environment:
    ```bash
    virtualenv env
    source env/bin/activate
    ```

5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. Run the main script:
    ```bash
    cd src
    python3 main.py
    ```

## Usage

The script requires a configuration file to specify various parameters for data processing and model training. Below is an example configuration file and an explanation of each section:

### Data Configuration

```yaml
data:
  source_url: "sogun3/uspollution"
  local_path: "data/us_pollution_data.csv"
  input_headers: ["year", "month", "day", "state", "city"]
  pollutant_headers: ["no2_mean", "o3_mean", "so2_mean", "co_mean"]
  test_size: 0.2
  random_state: 42
  specific_cities:
    - { state: "California", city: "Los Angeles" }
    - { state: "New York", city: "New York" }
```

- **source_url**: URL to download the dataset if it's not available locally.
- **local_path**: Path to the local CSV file containing the pollution data.
- **input_headers**: List of columns to be used as input features.
- **pollutant_headers**: List of pollutant columns to be used as target variables.
- **test_size**: Proportion of the dataset to include in the test split.
- **random_state**: Random seed for reproducibility.
- **specific_cities**: List of specific cities and states to filter the dataset.

### Regression Models

```yaml

models:
  regression:
    - type: "random_forest"
      hyperparameters:
        n_estimators: 100
        max_depth: 10
        random_state: 42
    - type: "linear_regression"
      hyperparameters: {}
    - type: "mlp"
      hyperparameters:
        hidden_layer_sizes: [100]
        max_iter: 200
        random_state: 42
    - type: "xgboost"
      hyperparameters:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        random_state: 42
```

- **type**: Specifies the type of regression model (e.g., "random_forest", "linear_regression", "mlp", "xgboost").
- **hyperparameters**: Dictionary of hyperparameters for each model.

### Time Series Models

```yaml
  time_series:
    - type: "arima"
      hyperparameters:
        p_range: [0, 4]
        d_range: [0, 3]
        q_range: [0, 4]
      specific_cities:
        - state: "California"
          city: "Los Angeles"
        - state: "New York"
          city: "New York"
    - type: "sarima"
      hyperparameters:
        p_range: [0, 4]
        d_range: [0, 3]
        q_range: [0, 4]
        sp_range: [0, 3]
        sd_range: [0, 2]
        sq_range: [0, 3]
      specific_cities:
        - state: "California"
          city: "Los Angeles"
        - state: "New York"
          city: "New York"
```

- **type**: Specifies the type of time series model (e.g., "arima", "sarima").
- **hyperparameters**: Dictionary of hyperparameters for each model, including ranges for parameters like p, d, q, sp, sd, and sq.
- **specific_cities**: Cities and states to focus the time series analysis on.

### Logging Configuration

```yaml

logging:
  version: 1
  disable_existing_loggers: False
  formatters:
    simple:
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    console:
      class: logging.StreamHandler
      formatter: simple
      level: INFO
    file:
      class: logging.FileHandler
      formatter: simple
      level: INFO
      filename: app.log
  root:
    handlers: [console, file]
    level: INFO
```

- **version**: Version of the logging configuration schema.
- **disable_existing_loggers**: Whether to disable existing loggers.
- **formatters**: Defines the format of log messages.
- **handlers**: Specifies where log messages are sent (e.g., console, file).
- **root**: Root logger configuration, combining all handlers and setting the logging level.

## Dataset

The dataset used for this analysis contains detailed information on air quality measurements across various locations and dates. Each record in the dataset includes multiple attributes that describe the pollutant levels and other related details. Below is a brief overview of the key columns present in the dataset:

- **State Code**: Numerical code representing the state.
- **County Code**: Numerical code representing the county.
- **Site Num**: Identification number of the monitoring site.
- **Address**: Address of the monitoring site.
- **State**: Name of the state.
- **County**: Name of the county.
- **City**: Name of the city.
- **Date Local**: Date of the measurement.
- **NO2 Units**: Measurement units for Nitrogen Dioxide (NO2).
- **NO2 Mean**: Mean value of NO2 on the given date.
- **NO2 1st Max Value**: Maximum value of NO2 recorded in a single hour.
- **NO2 1st Max Hour**: Hour of the day when the maximum NO2 value was recorded.
- **NO2 AQI**: Air Quality Index for NO2.
- **O3 Units**: Measurement units for Ozone (O3).
- **O3 Mean**: Mean value of O3 on the given date.
- **O3 1st Max Value**: Maximum value of O3 recorded in a single hour.
- **O3 1st Max Hour**: Hour of the day when the maximum O3 value was recorded.
- **O3 AQI**: Air Quality Index for O3.
- **SO2 Units**: Measurement units for Sulfur Dioxide (SO2).
- **SO2 Mean**: Mean value of SO2 on the given date.
- **SO2 1st Max Value**: Maximum value of SO2 recorded in a single hour.
- **SO2 1st Max Hour**: Hour of the day when the maximum SO2 value was recorded.
- **SO2 AQI**: Air Quality Index for SO2.
- **CO Units**: Measurement units for Carbon Monoxide (CO).
- **CO Mean**: Mean value of CO on the given date.
- **CO 1st Max Value**: Maximum value of CO recorded in a single hour.
- **CO 1st Max Hour**: Hour of the day when the maximum CO value was recorded.
- **CO AQI**: Air Quality Index for CO.

### Example Records

Here are some example records from the dataset:

| State Code | County Code | Site Num | Address                            | State    | County  | City    | Date Local | NO2 Units        | NO2 Mean | NO2 1st Max Value | NO2 1st Max Hour | NO2 AQI | O3 Units         | O3 Mean | O3 1st Max Value | O3 1st Max Hour | O3 AQI | SO2 Units        | SO2 Mean | SO2 1st Max Value | SO2 1st Max Hour | SO2 AQI | CO Units         | CO Mean  | CO 1st Max Value | CO 1st Max Hour | CO AQI |
|------------|-------------|----------|------------------------------------|----------|---------|---------|-------------|------------------|----------|------------------|------------------|---------|------------------|---------|------------------|-----------------|--------|------------------|----------|------------------|-----------------|--------|------------------|----------|------------------|-----------------|--------|
| 4          | 13          | 3002     | 1645 E ROOSEVELT ST-CENTRAL PHOENIX STN | Arizona | Maricopa | Phoenix | 2000-01-01  | Parts per billion | 19.042   | 49.0             | 19               | 46      | Parts per million | 0.0225  | 0.04             | 10              | 34     | Parts per billion | 3.0      | 9.0              | 21              | 13     | Parts per million | 1.146    | 4.2              | 21              |        |
| 4          | 13          | 3002     | 1645 E ROOSEVELT ST-CENTRAL PHOENIX STN | Arizona | Maricopa | Phoenix | 2000-01-01  | Parts per billion | 19.042   | 49.0             | 19               | 46      | Parts per million | 0.0225  | 0.04             | 10              | 34     | Parts per billion | 3.0      | 9.0              | 21              | 13     | Parts per million | 0.879    | 2.2              | 23              | 25     |
| 4          | 13          | 3002     | 1645 E ROOSEVELT ST-CENTRAL PHOENIX STN | Arizona | Maricopa | Phoenix | 2000-01-01  | Parts per billion | 19.042   | 49.0             | 19               | 46      | Parts per million | 0.0225  | 0.04             | 10              | 34     | Parts per billion | 2.975    | 6.6              | 23              |        | Parts per million | 1.146    | 4.2              | 21              |        |
| 4          | 13          | 3002     | 1645 E ROOSEVELT ST-CENTRAL PHOENIX STN | Arizona | Maricopa | Phoenix | 2000-01-01  | Parts per billion | 19.042   | 49.0             | 19               | 46      | Parts per million | 0.0225  | 0.04             | 10              | 34     | Parts per billion | 2.975    | 6.6              | 23              |        | Parts per million | 0.879    | 2.2              | 23              | 25     |
| 4          | 13          | 3002     | 1645 E ROOSEVELT ST-CENTRAL PHOENIX STN | Arizona | Maricopa | Phoenix | 2000-01-02  | Parts per billion | 22.958   | 36.0             | 19               | 34      | Parts per million | 0.0134  | 0.032            | 10              | 27     | Parts per billion | 1.958    | 3.0              | 22              | 4      | Parts per million | 0.850    | 1.6              | 23              |        |

## Data Engineering

The data engineering process involved several key steps to prepare and transform the dataset for analysis and modeling. First, the data was loaded from a CSV file, and the column names were standardized to lowercase snake case. Then, columns were renamed according to a specified dictionary, and only the necessary columns were filtered out. The date columns were converted to datetime format, allowing for the creation of additional time-related features such as year, month, day, day of the week, week of the year, and weekend indicators. Pollutant columns were normalized by subtracting the mean and dividing by the standard deviation. Duplicate rows were handled by grouping the data by specified columns and averaging the values. Further cleaning steps included label encoding categorical features, replacing infinite values with NaNs, filling NaNs with column means, and dropping any remaining NaN values and duplicates. This comprehensive data engineering process resulted in two distinct datasets, each tailored for specific types of analysis:

I. **The time series dataset** was created by retaining columns necessary for temporal analysis, including the date and newly created time-related features. This dataset was saved for time series analysis, enabling the examination of trends and patterns over time.

II. **The regression dataset** was tailored for predictive modeling by selecting input features and pollutant columns. This dataset was cleaned further by label encoding categorical variables and handling missing values, making it suitable for regression analysis and model training.

## Methodology

### I. Regression
We explored various machine learning models for regression tasks to predict pollutant levels, including:

- Linear Regression
- Multi-Layer Perceptron (MLP)
- Random Forest
- XGBoost

### II. Time Series

For time series analysis, we utilized ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal ARIMA) models to forecast future pollution levels.

Each model was trained on the training dataset and then used to make predictions about future pollutant levels, allowing us to evaluate their performance and effectiveness in capturing trends and patterns.

## Results for Regression Models

The table summarizes the performance of four different models (Random Forest, Linear Regression, MLP, XGBoost) across four different pollutants (NO2, O3, SO2, CO). The metrics used for evaluation are Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2).

| Pollutant | Model Name         | MSE    | MAE    | R2     |
|-----------|--------------------|--------|--------|--------|
| no2_mean  | random_forest      | 0.501  | 0.528  | 0.487  |
| no2_mean  | linear_regression  | 0.894  | 0.725  | 0.085  |
| no2_mean  | mlp                | 1.632  | 1.111  | -0.670 |
| no2_mean  | xgboost            | 0.376  | 0.444  | 0.616  |
| o3_mean   | random_forest      | 0.553  | 0.585  | 0.448  |
| o3_mean   | linear_regression  | 0.973  | 0.795  | 0.029  |
| o3_mean   | mlp                | 1.105  | 0.851  | -0.103 |
| o3_mean   | xgboost            | 0.479  | 0.545  | 0.522  |
| so2_mean  | random_forest      | 0.474  | 0.384  | 0.508  |
| so2_mean  | linear_regression  | 0.809  | 0.570  | 0.161  |
| so2_mean  | mlp                | 3.266  | 1.689  | -2.389 |
| so2_mean  | xgboost            | 0.457  | 0.382  | 0.526  |
| co_mean   | random_forest      | 0.439  | 0.428  | 0.544  |
| co_mean   | linear_regression  | 0.828  | 0.611  | 0.140  |
| co_mean   | mlp                | 1.156  | 0.876  | -0.201 |
| co_mean   | xgboost            | 0.398  | 0.408  | 0.586  |

1. Model Performance Comparison:
   - **XGBoost consistently outperforms other models** in terms of lower MSE and higher R2 values across all pollutants. This suggests that XGBoost is the most accurate model for predicting pollutant levels.
   - **Random Forest is the second-best performer**, showing good balance between MSE, MAE, and R2 values, but slightly less effective than XGBoost.
   - **Linear Regression performs poorly** compared to ensemble methods (Random Forest and XGBoost), indicating it may not capture the complexity in the data as well.
   - **MLP (Multi-Layer Perceptron) performs the worst**, with the highest MSE and lowest (often negative) R2 values, indicating it is not suitable for this dataset.

2. Pollutant-wise Observations:
   - **NO2 Mean:** XGBoost has the best performance (MSE: 0.376, R2: 0.616), while MLP has the worst (MSE: 1.632, R2: -0.670).
   - **O3 Mean:** XGBoost again leads (MSE: 0.479, R2: 0.522), with MLP showing poor performance (MSE: 1.105, R2: -0.103).
   - **SO2 Mean:** XGBoost and Random Forest are close in performance, with XGBoost slightly better (MSE: 0.457, R2: 0.526).
   - **CO Mean:** XGBoost performs best (MSE: 0.398, R2: 0.586), with MLP having the poorest results (MSE: 1.156, R2: -0.201).

### Plots for Predictions of the Best Model (XGBoost) for Train Data

The following plots illustrate the predictions made by the XGBoost model on the training data for each pollutant. These visualizations help in understanding how well the model captures the underlying patterns and relationships in the data.

I. CO Mean

![predictions_plot_bundled_co_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/dafbd507-7896-448f-91d6-a7bfa6db892d)

II. NO2 Mean

![predictions_plot_bundled_no2_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/266b49c5-f203-4f5e-8afa-ddc7ba7f7af1)

III. O3 Mean

![predictions_plot_bundled_o3_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/008c22b1-2a83-4c95-959b-13eaccdd43b0)

IV. SO2 Mean

![predictions_plot_bundled_so2_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/f9e35644-c690-4e25-bda3-5c3d60d42061)

## Analysis of Actual vs Predicted CO Mean in New York, NY

![predictions_plot_New York_New York_co_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/c7ad16f6-295f-4d28-b1d4-afcca2064c0d)

The above graph illustrates the comparison between actual and predicted CO mean levels over time for New York City, NY. The plot includes actual historical data, model predictions for the same period, and future predictions extending beyond the historical dataset.

### Observations

1. **Trends Over Time:**
   - The actual CO mean levels (blue bars) show a decreasing trend over the observed period, indicating an overall improvement in air quality in terms of CO levels.
   - The predicted CO mean levels (orange bars) closely follow the actual data, demonstrating the model's effectiveness in capturing the trend and variations in CO levels.

2. **Model Performance:**
   - The prediction bars (orange) generally align well with the actual bars (blue), suggesting that the model has good accuracy for historical data.
   - The future predicted values (green bars) continue the downward trend, indicating that the model anticipates further improvements in CO levels.

3. **Trend Line:**
   - The red dashed line represents the trend over the entire period, reinforcing the observed decrease in CO levels.

### Insights

1. **Accuracy and Reliability:**
   - The model demonstrates high accuracy, as evidenced by the close alignment between the actual and predicted values for the historical period.
   - This accuracy instills confidence in the model's future predictions.

2. **Future Predictions:**
   - The predicted future values suggest continued improvement in CO levels. If the model's assumptions hold, we can expect further reductions in CO pollution in New York City.
   - Policy makers and environmental agencies can use these predictions for planning and implementing further air quality improvements.

3. **Model Application:**
   - This analysis is useful for urban planners, environmentalists, and public health officials. Accurate predictions help in understanding the impact of current policies and in making informed decisions for future interventions.
  
### Other Pollutants

I. NO2 Mean

![predictions_plot_New York_New York_no2_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/9d17a12d-79e2-4e31-b082-047e2764b970)

II. O3 Mean

![predictions_plot_New York_New York_o3_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/c5b26e6e-be02-419b-9776-51199895bcfa)

III. SO2 Mean

![predictions_plot_New York_New York_so2_mean](https://github.com/djeada/Kaggle-US-Pollution/assets/37275728/3c25451d-360d-436d-99b5-edb41549e254)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
