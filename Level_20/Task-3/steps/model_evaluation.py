from zenml.steps import step
from sklearn.base import BaseEstimator
from evaluation.metrics import Evaluation
import pandas as pd

@step
def evaluate_models(data: dict, models: dict) -> dict:
    print("Evaluating models...")
    X_test = data["X_test"]
    y_test = data["y_test"]
    lr_model = models["lr_model"]
    rf_model = models["rf_model"]

    evaluation = Evaluation()
    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    lr_metrics = {
        "MSE": evaluation.mse(y_test, y_pred_lr),
        "RMSE": evaluation.rmse(y_test, y_pred_lr),
        "R2": evaluation.r2(y_test, y_pred_lr)
    }

    rf_metrics = {
        "MSE": evaluation.mse(y_test, y_pred_rf),
        "RMSE": evaluation.rmse(y_test, y_pred_rf),
        "R2": evaluation.r2(y_test, y_pred_rf)
    }

    print("Models evaluated successfully.")
    return {
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics
    }
