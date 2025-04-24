import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import os
from datetime import datetime

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

def generate_data():
    """Generate synthetic data for demonstration."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 5 * X[:, 2] + np.random.randn(100) * 0.1
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    df['target'] = y
    return df

def preprocess_data(df):
    """Preprocess the data."""
    X = df.drop('target', axis=1)
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def main():
    """Main pipeline function."""
    
    experiment_name = "linear_regression_tracking"
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass  
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
   
        mlflow.sklearn.autolog()
        
        start_time = datetime.now()
        mlflow.log_param("start_time", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        print("Step 1: Generating data...")
        df = generate_data()
        mlflow.log_param("data_shape", str(df.shape))
        
        print("Step 2: Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
    
        print("Step 3: Training model...")
        model = train_model(X_train, y_train)
        
        coef_dict = {f"coef_{i}": coef for i, coef in enumerate(model.coef_)}
        for name, value in coef_dict.items():
            mlflow.log_param(name, value)
        mlflow.log_param("intercept", model.intercept_)
        
        print("Step 4: Evaluating model...")
        mse, r2 = evaluate_model(model, X_test, y_test)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log end time and duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        mlflow.log_param("end_time", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_metric("duration_seconds", duration)
        
        # Print results
        print(f"\nModel Training Complete!")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Model Coefficients: {model.coef_}")
        print(f"Model Intercept: {model.intercept_:.4f}")
        
    print("\nAll steps completed successfully!")
    print("To view the results in MLflow UI, run: mlflow ui")
    print("Then open your browser to: http://localhost:5000")

if __name__ == "__main__":
    main()