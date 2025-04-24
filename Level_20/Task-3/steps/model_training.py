from zenml.steps import step
from sklearn.base import BaseEstimator
from models.model_dev import LinearRegressionModel, RandomForestModel
import pandas as pd

@step
def train_models(data: dict) -> dict:
    print("Training models...")
    X_train = data["X_train"]
    y_train = data["y_train"]
    lr_model = LinearRegressionModel().train(X_train, y_train)
    rf_model = RandomForestModel().train(X_train, y_train)
    print("Models trained successfully.")
    return {
        "lr_model": lr_model,
        "rf_model": rf_model
    }
