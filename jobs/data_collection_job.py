from dagster import job
from weather_project_dagster.assets.fetch_hourly_sample import fetch_hourly_sample
from weather_project_dagster.assets.collect_data import setup_directories, process_hourly_sample

@job
def data_collection_job():
    """Job для сбора и обработки часовых метеорологических данных"""
    sample = fetch_hourly_sample()
    setup_directories()
    process_hourly_sample(sample)