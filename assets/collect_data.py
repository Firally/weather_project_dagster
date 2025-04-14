import os
import pickle
import numpy as np
import pandas as pd
from dagster import op, get_dagster_logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

DATA_DIR = "./data"
MODEL_FILE = f"{DATA_DIR}/models.pkl"
TRAINING_DATA_FILE = f"{DATA_DIR}/training_data.csv"
ERROR_FILE = f"{DATA_DIR}/error_samples.csv"
THRESHOLD = 0.5  # Порог для стандартного отклонения предсказаний

@op
def setup_directories() -> None:
    """Создает необходимые директории для хранения данных и моделей"""
    logger = get_dagster_logger()
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Создана директория: {DATA_DIR}")

@op
def process_hourly_sample(sample: pd.DataFrame) -> None:
    """
    Оценивает сэмпл, используя ensemble предсказаний моделей.
    Если стандартное отклонение предсказаний больше THRESHOLD или моделей нет,
    сэмпл сохраняется в обучающую выборку.
    """
    logger = get_dagster_logger()
    
    if sample.empty:
        logger.warning("Получен пустой сэмпл, пропускаем обработку")
        return
    
    # Проверяем наличие моделей
    if not os.path.exists(MODEL_FILE):
        logger.info("Модели не найдены, сохраняем сэмпл по-умолчанию.")
        _save_sample(sample)
        return
    
    # Проверяем критерий для сохранения сэмпла
    if _should_save_sample(sample):
        _save_sample(sample)
        logger.info("Сэмпл удовлетворяет условию, сохранён.")
    else:
        logger.info("Сэмпл не соответствует порогу, не сохранён.")

def _should_save_sample(sample: pd.DataFrame) -> bool:
    """Определяет, нужно ли сохранять сэмпл на основе голосования моделей"""
    logger = get_dagster_logger()
    
    # Загружаем модели
    with open(MODEL_FILE, "rb") as f:
        models = pickle.load(f)
    
    features = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
        "precipitation", "rain", "snow_depth", "pressure_msl", "wind_direction_10m"
    ]
    
    # Проверяем, что все нужные фичи есть в сэмпле
    missing_features = set(features) - set(sample.columns)
    if missing_features:
        logger.warning(f"Отсутствуют фичи: {missing_features}, сохраняем сэмпл в файл ошибок.")
        _error_save_sample(sample)
        return False
    
    # Проверяем на наличие NaN в данных
    X_sample = sample[features]
    if X_sample.isna().any().any():
        logger.warning(f"Обнаружены NaN значения в данных, сохраняем сэмпл в файл ошибок.")
        _error_save_sample(sample)
        return False
    
    predictions = []
    
    # Собираем предсказания от всех моделей
    for model_name, model in models.items():
        try:
            predictions.append(model.predict(X_sample))
        except Exception as e:
            logger.error(f"Ошибка предсказания модели {model_name}: {str(e)}")
            _error_save_sample(sample)
            return False  # В случае ошибки предсказания не сохраняем сэмпл
    
    # Вычисляем стандартное отклонение предсказаний
    predictions = np.vstack(predictions)
    std_val = np.std(predictions, axis=0)[0]
    logger.info(f"Стандартное отклонение предсказаний: {std_val:.4f}")
    
    return std_val > THRESHOLD

def _save_sample(sample: pd.DataFrame) -> None:
    """Сохраняет сэмпл в файл обучающих данных"""
    logger = get_dagster_logger()
    
    # Сохраняем с временной меткой
    sample['saved_at'] = pd.Timestamp.now()
    
    if os.path.exists(TRAINING_DATA_FILE):
        sample.to_csv(TRAINING_DATA_FILE, mode="a", header=False, index=False)
        logger.info(f"Сэмпл добавлен к существующей обучающей выборке: {TRAINING_DATA_FILE}")
    else:
        sample.to_csv(TRAINING_DATA_FILE, mode="w", header=True, index=False)
        logger.info(f"Создан новый файл обучающей выборки: {TRAINING_DATA_FILE}")

def _error_save_sample(sample: pd.DataFrame) -> None:
    """Сохраняет сэмпл в файл ошибок"""
    logger = get_dagster_logger()
    
    # Сохраняем с временной меткой
    sample['saved_at'] = pd.Timestamp.now()
    
    if os.path.exists(ERROR_FILE):
        sample.to_csv(ERROR_FILE, mode="a", header=False, index=False)
        logger.info(f"Сэмпл добавлен к существующему файлу ошибок: {ERROR_FILE}")
    else:
        sample.to_csv(ERROR_FILE, mode="w", header=True, index=False)
        logger.info(f"Создан новый файл ошибок: {ERROR_FILE}")