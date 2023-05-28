from sklearn.linear_model import Lasso as SkLasso
import joblib

from src.models.base_model import BaseModel


class Lasso(BaseModel):
    def __init__(self):
        self.model = SkLasso()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename):
        joblib.dump(self.model, filename)

