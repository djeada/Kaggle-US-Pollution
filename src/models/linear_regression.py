import joblib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import GridSearchCV

from src.models.base_model import BaseModel


class LinearRegression(BaseModel):
    """
    Linear Regression implementation using sklearn.
    """

    def __init__(
        self,
        parameters={
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "copy_X": [True, False],
        },
    ):
        linear_regression = LR()
        self.model = GridSearchCV(
            linear_regression, parameters, verbose=1, scoring="r2"
        )

    def fit(self, x, y):
        """
        Train the model on the given data.
        :param x: The input data.
        :param y: The output data.
        :return: The trained model.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Predict the labels for the given data.
        :param x: The input data.
        :return: The predicted labels.
        """
        return self.model.predict(x)

    def save(self, path):
        """
        Serialize the model to the given path.
        :param path: The path to save the model to.
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load the model from the given path.
        :param path: The path to load the model from.
        """
        self.model = joblib.load(path)
