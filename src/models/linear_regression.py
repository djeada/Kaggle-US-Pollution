from sklearn.linear_model import LinearRegression as SkLinearRegression
import joblib

from src.models.base_model import BaseModel


class LinearRegression(BaseModel):
    def __init__(self):
        self.model = SkLinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename):
        joblib.dump(self.model, filename)



