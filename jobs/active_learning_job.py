from dagster import job
from assets.model_retraining import retrain_models

@job
def active_learning_job():
    """Job для дообучения моделей"""
    retrain_models()