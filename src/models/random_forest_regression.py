from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
import joblib

from src.models.base_model import BaseModel


class RandomForest(BaseModel):
    def __init__(self):
        self.model = SkRandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename):
        joblib.dump(self.model, filename)

