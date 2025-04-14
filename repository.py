from dagster import repository, schedule

from jobs.active_learning_job import active_learning_job
from jobs.data_collection_job import data_collection_job

@schedule(cron_schedule="0 * * * *", job=data_collection_job, execution_timezone="UTC")
def hourly_schedule():
    """Расписание для запуска data_collection_job каждый час"""
    return {}

@schedule(cron_schedule="0 0 * * 2", job=active_learning_job, execution_timezone="UTC")
def weekly_schedule():
    """Расписание для запуска active_learning_job каждый вторник в полночь"""
    return {}

@repository
def my_dagster_repository():
    """Репозиторий Dagster с job'ами и расписаниями"""
    return [
        active_learning_job,
        data_collection_job,
        hourly_schedule,
        weekly_schedule,
    ]