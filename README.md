# Kaggle U.S. Pollution
Exploration and modeling of U.S. Pollution dataset from Kaggle.

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

## Dataset

Our dataset includes historical AQI data with key measurements for pollutants such as Nitrogen Dioxide (NO2), Ozone (O3), Sulfur Dioxide (SO2), and Carbon Monoxide (CO). The data also includes information about the year, month, and location (city and state) of each measurement.

## Methodology

We explore different machine learning models for this task, including:

- Linear Regression
- Lasso Regression
- Random Forest
- XGBoost

Each model is trained on our dataset and then used to make predictions about future AQI levels. The performance of the models is evaluated using metrics like R2 Score, Root Mean Squared Error (RMSE), and Normalized RMSE.

## Findings

Our experiments suggest that while the machine learning models are able to fit the data to some extent, the predictive power is somewhat limited. The R2 scores range from negative values for Linear and Lasso Regression, indicating they perform worse than a horizontal line, to approximately 0.20 for Random Forest and 0.24 for XGBoost.

The RMSE values range from 11.22 for Linear Regression, 11.19 for Lasso, down to 9.29 for Random Forest and 8.82 for XGBoost. Lower RMSE values indicate better fit, so Random Forest and XGBoost perform better in this aspect, but there is still room for improvement.

Based on these findings, the Random Forest and XGBoost models may be used for preliminary forecasting, but their reliability is still questionable. The data may have too much randomness and might lack clear trends which are essential for reliable time-series predictions. 

Further investigation is required to identify other features or different model architectures that could enhance the performance.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
