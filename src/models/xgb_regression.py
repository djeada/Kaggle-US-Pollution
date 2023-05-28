from xgboost import XGBRegressor as SkXGBRegressor
import joblib

from src.models.base_model import BaseModel


class XGB(BaseModel):
    def __init__(self):
        self.model = SkXGBRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename):
        joblib.dump(self.model, filename)
