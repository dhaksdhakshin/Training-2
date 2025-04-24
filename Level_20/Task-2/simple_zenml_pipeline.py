import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
from zenml import pipeline, step
from typing import Tuple, Dict

@step
def ingest_data():
    """Data ingestion step."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 5 * X[:, 2] + np.random.randn(100) * 0.1
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    df['target'] = y
    return df

@step
def train_evaluate():
    """Combined training and evaluation step."""
    # Generate data (simplified for troubleshooting)
    df = ingest_data()
    
    # Process data
    X = df.drop('target', axis=1)
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog()
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Log parameter
    mlflow.log_param("model_type", "LinearRegression")
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {"mse": mse, "r2": r2}

@pipeline
def simple_pipeline():
    train_evaluate()

if __name__ == "__main__":
    simple_pipeline()