from zenml.steps import step

@step
def deployment_trigger(r2_score: float) -> bool:
    """Determines if the model meets the deployment threshold."""
    deploy = r2_score >= 0.7
    print(f"Deploy model? {'Yes' if deploy else 'No'}")
    return deploy