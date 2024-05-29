# Kaggle U.S. Pollution
To analyze trends, predict future pollution levels, and identify factors influencing air quality in different regions of the United States using advanced data engineering and machine learning techniques.

![dataset-cover](https://user-images.githubusercontent.com/37275728/198901751-79133990-37e2-4832-a698-3697dce69f50.jpeg)

## Introduction

> This dataset deals with pollution in the U.S. Pollution in the U.S. has been well documented by the U.S. EPA but it is a pain to download all the data and arrange them in a format that interests data scientists. Hence I gathered four major pollutants (Nitrogen Dioxide, Sulphur Dioxide, Carbon Monoxide and Ozone) for every day from 2000 - 2016 and place them neatly in a CSV file. 

<a href="https://www.kaggle.com/datasets/sogun3/uspollution">Read more.</a>

## Installation

Follow the steps:

- Download this repository: 
 
 ```bash 
 git clone https://github.com/djeada/kaggle-us-pollution.git
 ```
 
- Install <i>virtualenv</i> (if it's not already installed).
- Open the terminal from the project directory and run the following commands:

```bash
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
cd src
python3 main.py
```

Data Engineering

    Data Cleaning and Preprocessing: Handle missing values, normalize data, and manage temporal and spatial granularity.
    Feature Engineering: Create relevant features such as average pollutant levels over time, seasonal trends, and geographical clustering.

Machine Learning Models

 Exploratory Data Analysis (EDA)
 Visualize trends and patterns using time series plots, heatmaps, and geographical maps.
 Identify correlations between different pollutants and external factors.

Predictive Modeling
 Time Series Analysis: Utilize ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal ARIMA) models to forecast future pollution levels.
 Regression Models: Apply linear regression, decision trees, and random forests to understand the impact of various factors on pollution levels.
 Neural Networks: Implement LSTM (Long Short-Term Memory) networks for advanced time series forecasting.

Model Evaluation

    Training and Validation: Split the data into training and validation sets using techniques such as k-fold cross-validation to ensure robust model performance.
    Metrics: Evaluate models using metrics like RMSE (Root Mean Squared Error) for regression tasks, accuracy and F1 score for classification tasks, and silhouette score for clustering tasks.
    Model Interpretation: Use SHAP (SHapley Additive exPlanations) values to interpret the impact of different features on the model’s predictions.

Expected Outcomes

    Trends and Insights: Detailed insights into pollution trends over the years, including seasonal variations and geographical hotspots.
    Predictive Insights: Reliable forecasts of pollution levels, helping in proactive measures for pollution control.
    Policy Recommendations: Data-driven recommendations for policymakers to mitigate pollution based on the identified key factors.

Assessment

    Model Performance: Assess models based on prediction accuracy, interpretability, and computational efficiency.
    Real-world Applicability: Evaluate the practical applicability of the model predictions in real-world scenarios by comparing them with actual observed data.


## Dataset

Our dataset includes historical AQI data with key measurements for pollutants such as Nitrogen Dioxide (NO2), Ozone (O3), Sulfur Dioxide (SO2), and Carbon Monoxide (CO). The data also includes information about the year, month, and location (city and state) of each measurement.

## Methodology

We explore different machine learning models for this task, including:

- Linear Regression
- Lasso Regression
- Random Forest
- XGBoost

Each model is trained on our dataset and then used to make predictions about future AQI levels. The performance of the models is evaluated using metrics like R2 Score, Root Mean Squared Error (RMSE), and Normalized RMSE.

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
