import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import step

@step
def data_ingestion() -> (
    pd.DataFrame,  
    pd.DataFrame,
    pd.Series,    
    pd.Series     
):
    """Ingests and splits the dataset."""
   
    data = pd.read_csv("data/data.csv")
    
    X = data.drop(columns=["target"])
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
