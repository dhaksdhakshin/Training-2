from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train) -> BaseEstimator:
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train) -> BaseEstimator:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

class RandomForestModel(Model):
    def train(self, X_train, y_train) -> BaseEstimator:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model
