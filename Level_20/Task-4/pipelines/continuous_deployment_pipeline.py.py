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
    # Step 1: Ingest data
    X_train, X_test, y_train, y_test = data_ingestion()
    
    # Step 2: Train the model
    model = model_training(X_train, y_train)
    
    # Step 3: Evaluate the model
    r2_score = model_evaluation(model, X_test, y_test)
    
    # Step 4: Trigger deployment decision
    deploy = deployment_trigger(r2_score)
    
    # Step 5: Deploy the model (conditional)
    model_deployer(deploy=deploy, model=model)