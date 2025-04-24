import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
from zenml import pipeline, step
from zenml.client import Client

# Get the client and check if experiment_tracker exists
client = Client()
active_stack = client.active_stack

# Define step decorator with or without experiment tracker
def get_step_decorator(with_tracker=False):
    if with_tracker and active_stack.experiment_tracker:
        return lambda func: step(experiment_tracker=active_stack.experiment_tracker.name)(func)
    else:
        return step

# Use the helper function to conditionally apply experiment tracker
@get_step_decorator()
def ingest_data():
    """Data ingestion step."""
    # Replace this with your actual data loading code
    # For this example, we'll generate some synthetic data
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
    # Enable MLflow autologging for scikit-learn
    mlflow.sklearn.autolog()
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # You can also manually log additional parameters if needed
    if active_stack.experiment_tracker:
        mlflow.log_param("model_type", "LinearRegression")
    
    return model

@get_step_decorator(with_tracker=True)
def evaluate_model(model, X_test, y_test):
    """Model evaluation step with MLflow metric logging."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics to MLflow if experiment tracker exists
    if active_stack.experiment_tracker:
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
    
    # Print metrics to console anyway
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Return metrics as dictionary
    return {"mse": mse, "r2": r2}

@pipeline
def model_pipeline():
    """Main model training and evaluation pipeline."""
    df = ingest_data()
    X_train, X_test, y_train, y_test = process_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)