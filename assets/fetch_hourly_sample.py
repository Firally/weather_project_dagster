import requests
import pandas as pd
from dagster import op, get_dagster_logger
from datetime import datetime, timedelta

@op
def fetch_hourly_sample() -> pd.DataFrame:
    """Получение текущих часовых метеорологических данных"""
    logger = get_dagster_logger()
    
    # Используем текущие координаты и временной диапазон для запроса
    latitude = 68.36
    longitude = 18.82
    
    # Получаем данные за последний час
    now = datetime.now() - timedelta(days=30)
    start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    
    params = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "precipitation", "rain", "snow_depth", "et0_fao_evapotranspiration",
        "pressure_msl", "wind_direction_10m"
    ]
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params_str = ",".join(params)
    url = (
        f"{base_url}?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={params_str}&timezone=Europe/Berlin"
    )
    
    logger.info(f"Запрашиваем данные с URL: {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["hourly"])
            df["datetime"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"])
            # Берем только последний час данных
            df = df.iloc[-1:].copy()
            logger.info(f"Получен сэмпл данных: {df.shape}")
            return df
        else:
            logger.error(f"Ошибка запроса: {response.status_code} - {response.text}")
            # Возвращаем пустой DataFrame с правильными колонками в случае ошибки
            return pd.DataFrame(columns=params + ["datetime"])
    except Exception as e:
        logger.error(f"Ошибка при запросе данных: {str(e)}")
        return pd.DataFrame(columns=params + ["datetime"])