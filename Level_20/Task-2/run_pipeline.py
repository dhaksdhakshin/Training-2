from pipeline2 import model_pipeline
import sys

if __name__ == "__main__":
    try:
        pipeline_instance = model_pipeline()
        pipeline_instance.run()
        
        print("\nPipeline completed successfully!")
        print("To view results in the MLflow UI, run: mlflow ui")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        
        from zenml.client import Client
        client = Client()
        if not client.active_stack:
            print("\nNo active ZenML stack found. Try running:")
            print("zenml init")
            print("zenml stack register default_stack -a default -o default")
            print("zenml experiment-tracker register mlflow_tracker --flavor=mlflow")
            print("zenml stack update default_stack -e mlflow_tracker")
        sys.exit(1)
