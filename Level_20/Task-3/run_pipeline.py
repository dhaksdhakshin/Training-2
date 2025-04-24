import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelines.model_comparison_pipeline import model_comparison_pipeline

if __name__ == "__main__":
    print("Running pipeline...")
    pipeline_run = model_comparison_pipeline()
    print(f"Pipeline run completed: {pipeline_run}")
