from zenml.steps import step
import pandas as pd
from sklearn.model_selection import train_test_split

@step
def load_data() -> dict:
    print("Loading data...")
    data = pd.read_csv('data/your_dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data loaded successfully.")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
