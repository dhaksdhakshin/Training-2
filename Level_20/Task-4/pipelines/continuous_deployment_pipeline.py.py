from zenml.pipelines import pipeline

@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
    data_ingestion,
    model_training,
    model_evaluation,
    deployment_trigger,
    model_deployer
):
    """Defines the continuous deployment pipeline."""

    X_train, X_test, y_train, y_test = data_ingestion()
    
    model = model_training(X_train, y_train)
    
    r2_score = model_evaluation(model, X_test, y_test)
    
    deploy = deployment_trigger(r2_score)
    
    model_deployer(deploy=deploy, model=model)
