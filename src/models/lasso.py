import joblib
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.model_selection import GridSearchCV

from src.models.base_model import BaseModel


class Lasso(BaseModel):
    """
    Lasso implementation using sklearn.
    """

    def __init__(self, parameters={"alpha": [0.02, 0.024, 0.025, 0.026, 0.03]}):
        lasso = SklearnLasso()
        self.model = GridSearchCV(lasso, parameters, verbose=1, scoring="r2")

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
