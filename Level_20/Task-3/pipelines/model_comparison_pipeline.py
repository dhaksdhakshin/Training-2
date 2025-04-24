from zenml.pipelines import pipeline
from steps.data_loading import load_data
from steps.model_training import train_models
from steps.model_evaluation import evaluate_models

@pipeline
def model_comparison_pipeline():
    print("Starting pipeline...")
    data = load_data()
    models = train_models(data)
    metrics = evaluate_models(data, models)
    print("Pipeline completed successfully.")
    return metrics
