import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from zenml import pipeline, step
from typing import Tuple, Dict, Any

os.environ["ZENML_ANALYTICS_OPT_IN"] = "false"
os.environ["ZENML_CONFIG_PATH"] = os.path.join(os.path.expanduser("~"), ".zenml")

@step
def ingest_data() -> pd.DataFrame:
    """Ingest the diabetes dataset."""
    print("Ingesting data...")
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    
    df = X.copy()
    df['target'] = y
    
    return df

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing null values."""
    print("Cleaning data...")
    df_cleaned = df.dropna()
  
    print(f"Initial data shape: {df.shape}")
    print(f"Cleaned data shape: {df_cleaned.shape}")
    
    return df_cleaned

@step
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""
    print("Splitting data...")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a linear regression model."""
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model intercept: {model.intercept_}")
    print(f"Model coefficients: {model.coef_}")
    
    return model

@step
def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model with MSE and R2 scores."""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {"mse": mse, "r2": r2}

@pipeline
def diabetes_regression_pipeline():
    """Pipeline to train and evaluate a regression model on the diabetes dataset."""
    df = ingest_data()
    df_cleaned = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_cleaned)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    return metrics

if __name__ == "__main__":
    print("Starting diabetes regression pipeline...")
    results = diabetes_regression_pipeline()
    print(f"Pipeline completed successfully. Results: {results}") 