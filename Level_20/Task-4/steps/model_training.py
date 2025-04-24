from sklearn.ensemble import RandomForestRegressor
from zenml.steps import step
import pandas as pd

@step
def model_training(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestRegressor:
    """Trains a Random Forest Regressor."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model