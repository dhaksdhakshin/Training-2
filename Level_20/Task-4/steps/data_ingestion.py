import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import step

@step
def data_ingestion() -> (
    pd.DataFrame,  # X_train
    pd.DataFrame,  # X_test
    pd.Series,     # y_train
    pd.Series      # y_test
):
    """Ingests and splits the dataset."""
    # Load dataset (replace "data.csv" with your dataset path)
    data = pd.read_csv("data/data.csv")
    
    # Assume the target column is named "target"
    X = data.drop(columns=["target"])
    y = data["target"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test