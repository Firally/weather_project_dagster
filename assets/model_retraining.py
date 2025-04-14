import os
import pickle
import pandas as pd
from dagster import op, get_dagster_logger
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = "./data"
MODEL_FILE = f"{DATA_DIR}/models.pkl"
TRAINING_DATA_FILE = f"{DATA_DIR}/training_data.csv"
TESTING_DATA_FILE = f"{DATA_DIR}/testing_data.csv"
METRICS_FILE = f"{DATA_DIR}/model_metrics.csv"

def evaluate_model(model, X_test, y_test, model_name):
    """Вычисляет метрики для модели на тестовой выборке"""
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"model_name": model_name, "rmse": rmse, "r2": r2}

@op
def setup_directories() -> None:
    """Создает необходимые директории для хранения данных и моделей"""
    logger = get_dagster_logger()
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Создана директория: {DATA_DIR}")

@op
def retrain_models() -> dict:
    """
    Дообучает модели на накопленных данных и сохраняет их 
                            только если метрики улучшились
    """
    logger = get_dagster_logger()
    setup_directories()
    
    if not os.path.exists(TRAINING_DATA_FILE) or not os.path.exists(TESTING_DATA_FILE):
        logger.warning(f"Файлы данными не найдены: {TRAINING_DATA_FILE}, {TESTING_DATA_FILE}")
        return {}
    
    try:
        # Загружаем существующие модели, если они есть
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    existing_models = pickle.load(f)
                logger.info(f"Загружены существующие модели ({len(existing_models)} шт.)")
            except Exception as e:
                logger.warning(f"Не удалось загрузить существующие модели: {str(e)}")
        
        # Загружаем данные
        df = pd.read_csv(TRAINING_DATA_FILE)
        df_test = pd.read_csv(TESTING_DATA_FILE)
        logger.info(f"Загружены данные для обучения: {df.shape}")
        
        features = [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "precipitation", "rain", "snow_depth", "pressure_msl", "wind_direction_10m"
        ]
        target = "et0_fao_evapotranspiration"
        
        missing_cols = (set(features + [target]) - set(df.columns))
        if missing_cols:
            logger.error(f"Отсутствуют необходимые колонки: {missing_cols}")
            return {}
        
        X_train = df[features]
        y_train = df[target]

        X_test = df_test[features]
        y_test = df_test[target]
        logger.info(f"Формат данных для обучения: {X_train.shape}, {y_train.shape}")
        logger.info(f"Формат данных для тестирования: {X_test.shape}, {y_test.shape}")
        
        # Обучение моделей и оценка их метрик
        new_models = {}
        metrics = []
        models_improved = {}
        
        # Рассчитываем метрики существующих моделей для сравнения
        existing_metrics = {}
        for model_name, model in existing_models.items():
            try:
                _, existing_rmse, existing_r2 = evaluate_model(model, X_test, y_test, model_name).values()
                existing_metrics[model_name] = {"rmse": existing_rmse, "r2": existing_r2}
                logger.info(f"Текущие метрики {model_name}: RMSE={existing_rmse:.4f}, R²={existing_r2:.4f}")
            except Exception as e:
                logger.warning(f"Не удалось оценить существующую модель {model_name}: {str(e)}")
        
        # Линейная регрессия
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_metrics = evaluate_model(lr, X_test, y_test, "LinearRegression")
        metrics.append(lr_metrics)
        
        # Определяем, улучшилась ли линейная регрессия
        if "LinearRegression" not in existing_metrics:
            new_models["LinearRegression"] = lr
            models_improved["LinearRegression"] = True
            logger.info(f"Новая модель LinearRegression создана: RMSE={lr_metrics['rmse']:.4f}")
        elif lr_metrics['rmse'] < existing_metrics["LinearRegression"]["rmse"]:
            new_models["LinearRegression"] = lr
            models_improved["LinearRegression"] = True
            improvement = existing_metrics["LinearRegression"]["rmse"] - lr_metrics['rmse']
            logger.info(f"Модель LinearRegression улучшена: RMSE снизился на {improvement:.4f}")
        else:
            # Сохраняем старую модель
            new_models["LinearRegression"] = existing_models["LinearRegression"]
            models_improved["LinearRegression"] = False
            logger.info(f"Модель LinearRegression НЕ улучшена, сохраняем старую")
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_metrics = evaluate_model(rf, X_test, y_test, "RandomForest")
        metrics.append(rf_metrics)
        
        # Определяем, улучшился ли Random Forest
        if "RandomForest" not in existing_metrics:
            new_models["RandomForest"] = rf
            models_improved["RandomForest"] = True
            logger.info(f"Новая модель RandomForest создана: RMSE={rf_metrics['rmse']:.4f}")
        elif rf_metrics['rmse'] < existing_metrics["RandomForest"]["rmse"]:
            new_models["RandomForest"] = rf
            models_improved["RandomForest"] = True
            improvement = existing_metrics["RandomForest"]["rmse"] - rf_metrics['rmse']
            logger.info(f"Модель RandomForest улучшена: RMSE снизился на {improvement:.4f}")
        else:
            # Сохраняем старую модель
            new_models["RandomForest"] = existing_models["RandomForest"]
            models_improved["RandomForest"] = False
            logger.info(f"Модель RandomForest НЕ улучшена, сохраняем старую")
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_metrics = evaluate_model(xgb, X_test, y_test, "XGBoost")
        metrics.append(xgb_metrics)
        
        # Определяем, улучшился ли XGBoost
        if "XGBoost" not in existing_metrics:
            new_models["XGBoost"] = xgb
            models_improved["XGBoost"] = True
            logger.info(f"Новая модель XGBoost создана: RMSE={xgb_metrics['rmse']:.4f}")
        elif xgb_metrics['rmse'] < existing_metrics["XGBoost"]["rmse"]:
            new_models["XGBoost"] = xgb
            models_improved["XGBoost"] = True
            improvement = existing_metrics["XGBoost"]["rmse"] - xgb_metrics['rmse']
            logger.info(f"Модель XGBoost улучшена: RMSE снизился на {improvement:.4f}")
        else:
            # Сохраняем старую модель
            new_models["XGBoost"] = existing_models["XGBoost"]
            models_improved["XGBoost"] = False
            logger.info(f"Модель XGBoost НЕ улучшена, сохраняем старую")
        
        # Сохраняем обновленный набор моделей
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(new_models, f)
        logger.info(f"Модели сохранены в файл: {MODEL_FILE}")
        
        # Добавляем информацию об улучшении в метрики
        for i, metric in enumerate(metrics):
            model_name = metric["model_name"]
            metrics[i]["improved"] = models_improved.get(model_name, False)
        
        # Сохраняем метрики
        metrics_df = pd.DataFrame(metrics)
        metrics_df["timestamp"] = pd.Timestamp.now()
        
        if os.path.exists(METRICS_FILE):
            metrics_df.to_csv(METRICS_FILE, mode="a", header=False, index=False)
        else:
            metrics_df.to_csv(METRICS_FILE, mode="w", header=True, index=False)
        logger.info(f"Метрики сохранены в файл: {METRICS_FILE}")
        
        # Возвращаем метрики и информацию об улучшении
        return {
            model["model_name"]: {
                "rmse": model["rmse"], 
                "r2": model["r2"], 
                "improved": model["improved"]
            } for model in metrics
        }
    
    except Exception as e:
        logger.error(f"Ошибка при дообучении моделей: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}