from steps.data_ingestion import data_ingestion
from steps.model_training import model_training
from steps.model_evaluation import model_evaluation
from steps.deployment_trigger import deployment_trigger
from steps.model_deployer import model_deployer
from pipelines.continuous_deployment_pipeline import continuous_deployment_pipeline

if __name__ == "__main__":
    pipeline_instance = continuous_deployment_pipeline(
        data_ingestion=data_ingestion(),
        model_training=model_training(),
        model_evaluation=model_evaluation(),
        deployment_trigger=deployment_trigger(),
        model_deployer=model_deployer
    )
    
    pipeline_instance.run()
