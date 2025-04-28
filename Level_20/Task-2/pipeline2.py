import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
from zenml import pipeline, step
from zenml.client import Client

client = Client()
active_stack = client.active_stack

def get_step_decorator(with_tracker=False):
    if with_tracker and active_stack.experiment_tracker:
        return lambda func: step(experiment_tracker=active_stack.experiment_tracker.name)(func)
    else:
        return step

@get_step_decorator()
def ingest_data():
    """Data ingestion step."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 5 * X[:, 2] + np.random.randn(100) * 0.1
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    df['target'] = y
    return df

@step
def process_data(df: pd.DataFrame):
    """Data processing step."""
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@get_step_decorator(with_tracker=True)
def train_model(X_train, y_train):
    """Model training step with MLflow auto logging."""
  
    mlflow.sklearn.autolog()
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    if active_stack.experiment_tracker:
        mlflow.log_param("model_type", "LinearRegression")
    
    return model

@get_step_decorator(with_tracker=True)
def evaluate_model(model, X_test, y_test):
    """Model evaluation step with MLflow metric logging."""

    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if active_stack.experiment_tracker:
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {"mse": mse, "r2": r2}

@pipeline
def model_pipeline():
    """Main model training and evaluation pipeline."""
    df = ingest_data()
    X_train, X_test, y_train, y_test = process_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
