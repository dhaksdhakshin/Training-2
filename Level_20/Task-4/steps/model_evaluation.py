from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from zenml.steps import step
import pandas as pd

@step
def model_evaluation(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """Evaluates the model and computes the R2 score."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Model R2 Score: {r2}")
    return r2