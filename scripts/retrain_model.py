import argparse
import os
import logging
import json
import ast
from datetime import datetime
import numpy as np
import pandas as pd
import comet_ml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import (
    train_test_split, 
    RandomizedSearchCV, 
    cross_val_score, 
    GridSearchCV, 
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
from comet_ml import Experiment
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import joblib



# ————————————————————————————————
# Настройка логгера
# ————————————————————————————————

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

try:
    from hyperparam_tuning import tune_xgb, tune_xgb_for_recall
except ImportError:
    logger.warning("Модуль тюнинга недоступен. Используются базовые параметры.")
# ————————————————————————————————
# Конфигурация
# ————————————————————————————————

SECRETS_PATH = "/path/to/secrets.json"

if not os.path.exists(SECRETS_PATH):
    raise FileNotFoundError(f"Файл secrets.json не найден по пути {SECRETS_PATH}")

with open(SECRETS_PATH, "r") as f:
    secrets = json.load(f)

COMET_API_KEY = secrets.get("comet_api_key")
if not COMET_API_KEY:
    raise ValueError("Ключ 'comet_api_key' отсутствует в файле secrets.json")

DATASET_PATH = "/path/to/your.csv"
best_params_path = os.path.join(os.path.dirname(DATASET_PATH),
                                os.path.splitext(os.path.basename(DATASET_PATH))[0] + "_best_params.json")

SAMPLING_METHODS = {
    'none': None,
    'smote': SMOTE(random_state=42),
    'borderline': BorderlineSMOTE(random_state=42),
    'adasyn': ADASYN(random_state=42),
    'random_over': RandomOverSampler(random_state=42),
    'random_under': RandomUnderSampler(random_state=42)
}

MODEL_TYPES = {
    'rf': RandomForestClassifier,
    'xgb': XGBClassifier,
    'lgb': LGBMClassifier,
    'rfr': RandomForestRegressor  # NEW
}


# ————————————————————————————————
# Генерация уникальной директории вывода
# ————————————————————————————————

def generate_output_dir(dataset_path: str, base_output_dir: str = "retrained_results") -> str:
    filename = os.path.splitext(os.path.basename(dataset_path))[0]
    base_name = '_'.join(filename.split('_')[:-1]) if '_' in filename else filename
    safe_base_name = base_name.replace(' ', '_').replace('.', '_')
    output_dir = f"{base_output_dir}/{safe_base_name}"
    version = 1
    while os.path.exists(output_dir):
        version += 1
        output_dir = f"{base_output_dir}/{safe_base_name}_v{version}"
    return output_dir


OUTPUT_DIR = generate_output_dir(DATASET_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ————————————————————————————————
# Функции
# ————————————————————————————————

def load_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Загрузка датасета из {path}")
    df = pd.read_csv(path)
    df['detection_vector'] = df['detection_vector'].apply(lambda x: np.array(ast.literal_eval(x)))
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Подготовка данных...")
    X = np.stack(df['detection_vector'].values)
    y = df['category_present'].values
    return X, y


def get_best_params(model_type: str, best_params_path: Optional[str] = None) -> Dict[str, Any]:
    if best_params_path and os.path.exists(best_params_path):
        with open(best_params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        logger.info("Параметры загружены из файла")
        return params
    
    # Базовые параметры для каждой модели
    base_params = {
        'random_state': 42,
        'n_jobs': -1
    }
    
    if model_type == 'rf':
        return {**base_params, **{
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }}
    elif model_type == 'rfr':  # NEW
        return {**base_params, **{
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }}
    elif model_type == 'xgb':
        return {
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist'  # Для ускорения обучения
        }
    elif model_type == 'lgb':
        return {
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1  # Для подавления выводов
        }
    else:
        logger.warning(f"Неизвестная модель: {model_type}. Используются стандартные параметры.")
        return {}


def train_model(X_train, y_train, sampling_method, model_type, tune=False, 
               recall_opt=False, tune_method="random_search", 
               early_stopping=False, stopping_rounds=10,
               pretrained_model=None):  # Добавлен новый параметр
    """
    Обучение модели с возможностью дообучения предобученной модели,
    проверкой баланса классов и обработкой крайних случаев.
    
    Параметры:
        pretrained_model: Предобученная модель (опционально)
        ... остальные существующие параметры ...
    """
    logger.info(f"Обучение модели {model_type} с методом балансировки: {sampling_method}")
    
    # Проверка и обработка предобученной модели
    if pretrained_model:
        logger.info("Использование предобученной модели для дообучения")
        if not hasattr(pretrained_model, 'partial_fit'):
            logger.warning("Модель не поддерживает дообучение. Будет выполнено обучение с нуля.")
        else:
            try:
                pretrained_model.fit(X_train, y_train)
                return pretrained_model
            except Exception as e:
                logger.error(f"Ошибка дообучения: {str(e)}. Будет выполнено обучение с нуля.")
    
    # Проверка баланса классов
    class_counts = np.bincount(y_train)
    logger.info(f"Распределение классов: {dict(zip(np.unique(y_train), class_counts))}")
    
    # Проверка на единственный класс
    if len(class_counts) < 2:
        logger.warning("Обнаружен только один класс в данных. Сэмплинг не будет применен.")
        sampling_method = 'none'
    elif len(class_counts) > 1 and sampling_method != 'none':
        # Проверка необходимости сэмплинга (если дисбаланс < 10%, не применяем)
        imbalance_ratio = max(class_counts) / min(class_counts)
        if imbalance_ratio < 1.1:
            logger.info(f"Дисбаланс классов незначительный ({imbalance_ratio:.1f}x). Сэмплинг не применяется.")
            sampling_method = 'none'
    
    # Обработка балансировки классов
    if model_type == 'rfr':
        sampler = None
        X_res, y_res = X_train, y_train
    else:
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
        sampler = SAMPLING_METHODS.get(sampling_method.lower(), None)
        if sampler is not None:
            try:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                logger.info(f"Применен {sampling_method}. Новый размер данных: {len(X_res)}")
            except Exception as e:
                logger.error(f"Ошибка при сэмплинге: {str(e)}. Используются исходные данные.")
                X_res, y_res = X_train, y_train
        else:
            X_res, y_res = X_train, y_train

    # Тюнинг для XGBoost
    if tune and model_type == 'xgb':
        params_cache = os.path.join(OUTPUT_DIR, 
                                  f"best_params_{model_type}_{sampling_method}_{'recall' if recall_opt else 'f1'}.json")
        
        if os.path.exists(params_cache):
            with open(params_cache, 'r') as f:
                params = json.load(f)
        else:
            if tune_method == "grid_search":
                best_params = tune_xgb_for_recall(X_res, y_res) if recall_opt else tune_xgb(X_res, y_res)
            else:
                best_params = tune_xgb_advanced(
                    X_res, y_res, 
                    method=tune_method,
                    recall_opt=recall_opt,
                    early_stopping=early_stopping,
                    stopping_rounds=stopping_rounds
                )
            
            params = get_best_params(model_type, best_params_path)
            params.update(best_params)
            with open(params_cache, 'w') as f:
                json.dump(best_params, f)
    
    # Тюнинг для LightGBM
    elif tune and model_type == 'lgb':
        params_cache = os.path.join(OUTPUT_DIR, 
                                  f"best_params_{model_type}_{sampling_method}_{'recall' if recall_opt else 'f1'}.json")
        
        if os.path.exists(params_cache):
            with open(params_cache, 'r') as f:
                params = json.load(f)
        else:
            best_params = tune_lgbm(
                X_res, y_res,
                method=tune_method,
                recall_opt=recall_opt,
                early_stopping=early_stopping,
                stopping_rounds=stopping_rounds
            )
            
            params = get_best_params(model_type, best_params_path)
            params.update(best_params)
            with open(params_cache, 'w') as f:
                json.dump(best_params, f)
    
    # Тюнинг для регрессора
    elif tune and model_type == 'rfr':
        if tune_method == "random_search":
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [5, 10, 15, 20, 25, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]
            }
            model = RandomForestRegressor(random_state=42)
            search = RandomizedSearchCV(
                model, param_dist, n_iter=50, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            search.fit(X_res, y_res)
            return search.best_estimator_
        
        elif tune_method == "optuna":            
            class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
                pass
            
            class EarlyStoppingCallback:
                def __init__(self, early_stopping_rounds):
                    self.early_stopping_rounds = early_stopping_rounds
                    self._best_score = None
                    self._no_improvement_count = 0
                
                def __call__(self, study, trial):
                    if self._best_score is None:
                        self._best_score = study.best_value
                        return
                    
                    if study.best_value < self._best_score:
                        self._best_score = study.best_value
                        self._no_improvement_count = 0
                    else:
                        self._no_improvement_count += 1
                    
                    if self._no_improvement_count >= self._early_stopping_rounds:
                        raise EarlyStoppingExceeded()
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_float('max_features', 0.1, 0.9),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
                return -cross_val_score(
                    model, X_res, y_res,
                    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                ).mean()
            
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler
            )
            
            try:
                study.optimize(
                    objective,
                    n_trials=100,
                    callbacks=[EarlyStoppingCallback(stopping_rounds)] if early_stopping else None
                )
            except EarlyStoppingExceeded:
                logger.info(f"Ранняя остановка после {stopping_rounds} раундов без улучшений")
            
            best_model = RandomForestRegressor(**study.best_params, random_state=42)
            best_model.fit(X_res, y_res)
            
            best_params_path = os.path.join(OUTPUT_DIR, f"best_params_rfr_{sampling_method}.json")
            with open(best_params_path, 'w') as f:
                json.dump(study.best_params, f)
            
            return best_model
    
    params = params if tune else get_best_params(model_type, best_params_path)
    
    # Параметры ранней остановки
    if model_type in ['xgb', 'lgb'] and early_stopping:
        X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        if model_type == 'xgb':
            params['early_stopping_rounds'] = stopping_rounds
            params['eval_set'] = [(X_val, y_val)]
            params['eval_metric'] = 'logloss' if not recall_opt else 'recall'
        elif model_type == 'lgb':
            params['early_stopping_round'] = stopping_rounds
            params['eval_set'] = [(X_val, y_val)]
            params['eval_metric'] = 'binary_logloss' if not recall_opt else 'recall'
    
    # Создание модели
    if model_type == 'rf':
        model = RandomForestClassifier(**params)
    elif model_type == 'rfr':
        model = RandomForestRegressor(**params)
    elif model_type == 'xgb':
        model = XGBClassifier(**params)
    elif model_type == 'lgb':
        model = LGBMClassifier(**params)
    
    try:
        model.fit(X_res, y_res)
        logger.info("Модель успешно обучена")
        return model
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {str(e)}")
        raise

def tune_xgb_advanced(X, y, method="random_search", recall_opt=False, 
                     early_stopping=False, stopping_rounds=10):
    """Расширенный тюнинг для XGBoost с поддержкой разных методов"""
    if method == "grid_search":
        # Используем существующую функцию
        return tune_xgb_for_recall(X, y) if recall_opt else tune_xgb(X, y)
    
    scoring = 'recall' if recall_opt else 'f1'
    
    if method == "random_search":
        param_dist = {
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200, 300],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        model = XGBClassifier(objective='binary:logistic', random_state=42)
        search = RandomizedSearchCV(
            model, param_dist, n_iter=50, cv=3,
            scoring=scoring, n_jobs=-1, random_state=42
        )
        search.fit(X, y)
        return search.best_params_
    
    elif method == "optuna":        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'gamma': trial.suggest_float('gamma', 0, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary:logistic',
                'random_state': 42
            }
            model = XGBClassifier(**params)
            return cross_val_score(
                model, X, y, cv=3, scoring=scoring, n_jobs=-1
            ).mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        return study.best_params

def tune_lgbm(X, y, method="random_search", recall_opt=False,
              early_stopping=False, stopping_rounds=10):
    """Тюнинг для LightGBM"""
    scoring = 'recall' if recall_opt else 'f1'
    
    if method == "grid_search":
        param_grid = {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_samples': [10, 20]
        }
        model = LGBMClassifier(objective='binary', random_state=42)
        search = GridSearchCV(
            model, param_grid, cv=3,
            scoring=scoring, n_jobs=-1
        )
        search.fit(X, y)
        return search.best_params_
    
    elif method == "random_search":
        param_dist = {
            'num_leaves': [15, 31, 63, 127],
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200, 300],
            'min_child_samples': [5, 10, 20, 30],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        model = LGBMClassifier(objective='binary', random_state=42)
        search = RandomizedSearchCV(
            model, param_dist, n_iter=50, cv=3,
            scoring=scoring, n_jobs=-1, random_state=42
        )
        search.fit(X, y)
        return search.best_params_
    
    elif method == "optuna":        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary',
                'random_state': 42
            }
            model = LGBMClassifier(**params)
            return cross_val_score(
                model, X, y, cv=3, scoring=scoring, n_jobs=-1
            ).mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        return study.best_params

def save_predictions(df: pd.DataFrame, model, X_full: np.ndarray, output_dir: str, suffix: str, threshold: float) -> str:
    logger.info("Сохранение предсказаний...")
    if args.model == 'rfr':
        df[f'forest_pred_{suffix}'] = model.predict(X_full)
    else:
        df[f'forest_pred_{suffix}'] = model.predict_proba(X_full)[:, 1]
        df[f'forest_pred_binary_{suffix}'] = (df[f'forest_pred_{suffix}'] >= threshold).astype(int)
    output_path = os.path.join(output_dir, f"retrained_forest_predictions_{suffix}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Предсказания сохранены в {output_path}")
    return output_path


def evaluate_model(y_true: np.ndarray, y_score: np.ndarray, threshold: float, suffix: str) -> Dict[str, float]:
    logger.info(f"Оценка модели ({suffix})")
    
    if args.model == 'rfr':  # Режим регрессии
        metrics = {
            'mse': round(mean_squared_error(y_true, y_score), 6),
            'r2': round(r2_score(y_true, y_score), 6)
        }
        logger.info(f"MSE: {metrics['mse']}, R2: {metrics['r2']}")
    else:  # Классификация
        y_pred = (y_score >= threshold).astype(int)
        metrics = {
            'accuracy': round(accuracy_score(y_true, y_pred), 6),
            'precision': round(precision_score(y_true, y_pred, zero_division=0), 6),
            'recall': round(recall_score(y_true, y_pred, zero_division=0), 6),
            'f1': round(f1_score(y_true, y_pred, zero_division=0), 6),
            'roc_auc': round(roc_auc_score(y_true, y_score), 6) if len(np.unique(y_true)) > 1 else 0.5
        }
        logger.info(classification_report(y_true, y_pred, zero_division=0))
        logger.info(f"ROC AUC: {metrics['roc_auc']}")

    # Сохранение метрик
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{suffix}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Метрики сохранены в {metrics_path}")

    # Матрица ошибок только для классификации
    if args.model != 'rfr':
        cm = confusion_matrix(y_true, y_pred)
        cm_dict = {'tp': int(cm[1, 1]), 'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0])}
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{suffix}.json")
        with open(cm_path, 'w') as f:
            json.dump(cm_dict, f, indent=4)
        logger.info(f"Матрица ошибок сохранена в {cm_path}")

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if args.model == 'rfr':
        return 0.5  
    logger.info("Расчёт оптимального порога (F1-optimal)")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    return thresholds[np.argmax(f1_scores)]


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "Calibration Curve"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(title)
    plt.legend()
    plt.grid()
    return plt


def plot_feature_importance(model, feature_names: list, title: str = "Feature Importance", top_n: int = 15):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Модель не поддерживает feature_importances_ или coef_")
            return None

        idx = np.argsort(importances)[::-1][:top_n]
        sorted_features = [(feature_names[i], importances[i]) for i in idx]

        plt.figure(figsize=(12, 8))
        plt.barh([f[0] for f in sorted_features], [f[1] for f in sorted_features])
        plt.xlabel("Важность")
        plt.ylabel("Признаки")
        plt.title(title)
        plt.tight_layout()

        fig_path = os.path.join(OUTPUT_DIR, f"feature_importance_{title}.png")
        plt.savefig(fig_path)
        plt.close()

        logger.info(f"График важности признаков сохранён в {fig_path}")
        return fig_path
    except Exception as e:
        logger.error(f"Ошибка при построении важности признаков: {str(e)}")
        return None


def append_run_to_category_report(
    category_name: str,
    start_time: datetime,
    model_type: str,
    user_metrics: dict,
    optimal_metrics: dict,
    user_threshold: float,
    optimal_threshold: float,
    sampling_method: str,
    comment: str = ""
):
    try:
        safe_name = "".join([c if c.isalnum() or c == " " else "_" for c in category_name])
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{safe_name}.md")
        
        # Check if file exists and count lines if it does
        entry_number = 0
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Count non-header lines (assuming 3 header lines)
                entry_number = len([line for line in lines if line.startswith("|") and not line.startswith("| ---")]) - 3
        else:
            # Create new file with headers
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Сводная таблица применения моделей по категории: {category_name}\n\n")
                f.write(f"- Проведены различные стратегии балансировки классов данных, полученных с различными признаками, учтен пользовательский порог, применено сравнение прогонов и поддержка разных моделей: RandomForestClassifier, RandomForestRegressor, XGBoost, LightGBM\n\n")
                f.write(f"- В ходе анализа выполнено логирование в Comet ML, сохранение графиков важности признаков и других артефактов\n\n")
        
                # Table headers
                f.write("| № | Время | Threshold | Model | Sampling |")
                if model_type == 'rfr':
                    f.write(" MSE | R2 | Comments |\n")
                else:
                    f.write(" Accuracy | Precision | Recall | F1 | ROC AUC | Comments |\n")
                
                # Column separators
                f.write("|---|-------|-----------|-------|----------|")
                if model_type == 'rfr':
                    f.write("---------|---------|----------|\n")
                else:
                    f.write("----------|-----------|--------|----|---------|----------|\n")

        # Append new entries
        with open(report_path, "a", encoding="utf-8") as f:
            if model_type == 'rfr':
                row_user = (
                    f"| {entry_number + 1} "
                    f"| {start_time.strftime('%Y-%m-%d %H:%M')} "
                    f"| - "
                    f"| {model_type.upper()} "
                    f"| {sampling_method} "
                    f"| {user_metrics['mse']:.4f} "
                    f"| {user_metrics['r2']:.4f} "
                    f"| {comment} |\n"
                )
                row_optimal = (
                    f"| {entry_number + 2} "
                    f"| {start_time.strftime('%Y-%m-%d %H:%M')} "
                    f"| - "
                    f"| {model_type.upper()} "
                    f"| {sampling_method} "
                    f"| {optimal_metrics['mse']:.4f} "
                    f"| {optimal_metrics['r2']:.4f} "
                    f"| {comment} (optimal) |\n"
                )
            else:
                row_user = (
                    f"| {entry_number + 1} "
                    f"| {start_time.strftime('%Y-%m-%d %H:%M')} "
                    f"| {user_threshold:.4f} (user) "
                    f"| {model_type.upper()} "
                    f"| {sampling_method} "
                    f"| {user_metrics['accuracy']:.4f} "
                    f"| {user_metrics['precision']:.4f} "
                    f"| {user_metrics['recall']:.4f} "
                    f"| {user_metrics['f1']:.4f} "
                    f"| {user_metrics['roc_auc']:.4f} "
                    f"| {comment} |\n"
                )

                row_optimal = (
                    f"| {entry_number + 2} "
                    f"| {start_time.strftime('%Y-%m-%d %H:%M')} "
                    f"| {optimal_threshold:.4f} (optimal) "
                    f"| {model_type.upper()} "
                    f"| {sampling_method} "
                    f"| {optimal_metrics['accuracy']:.4f} "
                    f"| {optimal_metrics['precision']:.4f} "
                    f"| {optimal_metrics['recall']:.4f} "
                    f"| {optimal_metrics['f1']:.4f} "
                    f"| {optimal_metrics['roc_auc']:.4f} "
                    f"| {comment} (optimal threshold) |\n"
                )

            f.write(row_user)
            f.write(row_optimal)

        return report_path
        
    except Exception as e:
        logger.error(f"Ошибка при создании отчета: {str(e)}")
        return None

def setup_comet_experiment(category_name: str, model_type: str, sampling_method: str, feature_count: int, class_ratio: float):
    logger.info("Инициализация эксперимента в Comet ML")
    project_name = f"{category_name.replace(' ', '-')}-classification"
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name=project_name,
        workspace="lidiya-cutie",
        auto_param_logging=False,
        log_code=True
    )
    start_time = datetime.now()
    experiment.set_name(f"{category_name.replace(' ', '_')}_{start_time.strftime('%Y%m%d-%H%M%S')}")
    experiment.add_tag(category_name)
    experiment.add_tag(model_type)
    experiment.add_tag("clip")
    experiment.add_tag("retraining")
    experiment.add_tag(sampling_method)

    experiment.log_parameters({
        "dataset_path": DATASET_PATH,
        "sampling_method": sampling_method,
        "threshold": args.threshold,
        "category": category_name,
        "model_type": model_type,
        "feature_count": feature_count,
        "class_ratio": class_ratio,
        "test_size": 0.15,
        "random_state": 42
    })

    return experiment, start_time


# ————————————————————————————————
# Точка входа
# ————————————————————————————————

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Повторное обучение модели")
    parser.add_argument("--model", type=str, default="rf", 
                       choices=["rf", "xgb", "lgb", "rfr"], 
                       help="Тип модели: rf - RandomForest, xgb - XGBoost, lgb - LightGBM, rfr - RandomForestRegressor")
    parser.add_argument("--sampling", type=str, default="none", choices=SAMPLING_METHODS.keys(),
                       help="Метод балансировки классов")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Порог для бинаризации предсказаний")
    parser.add_argument("--tune", action="store_true", 
                       help="Включить перебор гиперпараметров")
    parser.add_argument("--tune-method", type=str, default="random_search",
                       choices=["random_search", "optuna", "grid_search"],
                       help="Метод тюнинга: random_search, optuna или grid_search")
    parser.add_argument("--pretrained-model", type=str, default=None,
                       help="Путь к файлу предобученной модели (.joblib)")
    parser.add_argument("--early-stopping", action="store_true",
                       help="Включить раннюю остановку для XGBoost/LightGBM")
    parser.add_argument("--stopping-rounds", type=int, default=10,
                       help="Количество раундов без улучшений для ранней остановки")
    parser.add_argument("--cross-validate", action="store_true",
                       help="Использовать кросс-валидацию вместо train-test split")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Количество фолдов для кросс-валидации")
    parser.add_argument("--recall-opt", action="store_true",
                       help="Оптимизировать под Recall вместо F1-score")
    parser.add_argument("--ctgr-name", type=str, default="",
                       help="Имя категории для логирования")
    parser.add_argument("--comment", type=str, default="", help="Комментарий для прогона")
    args = parser.parse_args()
    
    logger.info("Параметры запуска:")
    logger.info("\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    # Добавленная проверка предобученной модели ДО загрузки данных
    if args.pretrained_model:
        if not os.path.exists(args.pretrained_model):
            raise FileNotFoundError(f"Файл модели не найден: {args.pretrained_model}")
        if not args.pretrained_model.endswith(('.joblib', '.pkl')):
            logger.warning("Предобученная модель должна быть в формате .joblib или .pkl")

    try:
        df = load_dataset(DATASET_PATH)
        X, y = prepare_data(df)
        
        # Модифицированный блок загрузки предобученной модели
        pretrained_model = None
        if args.pretrained_model:
            try:
                pretrained_model = joblib.load(args.pretrained_model)
                logger.info(f"Загружена предобученная модель: {type(pretrained_model).__name__}")
                
                # Проверка совместимости модели
                expected_type = MODEL_TYPES.get(args.model)
                if expected_type and not isinstance(pretrained_model, expected_type):
                    logger.warning(f"Тип модели не совпадает! Ожидается {expected_type}, получен {type(pretrained_model)}")
                    pretrained_model = None
                
                # Проверка размерности данных
                if pretrained_model and hasattr(pretrained_model, 'n_features_in_'):
                    if pretrained_model.n_features_in_ != X.shape[1]:
                        logger.warning(f"Несовпадение размерности! Ожидается {pretrained_model.n_features_in_} признаков, получено {X.shape[1]}")
                        pretrained_model = None
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {str(e)}")
                pretrained_model = None

        # Инициализация Comet
        experiment, start_time = setup_comet_experiment(
            args.ctgr_name, 
            args.model, 
            args.sampling, 
            X.shape[1], 
            sum(y) / len(y) if len(y) > 0 else 0
        )

        # Добавлено: Логирование всех параметров в начале эксперимента
        experiment.log_parameters(vars(args))
        if pretrained_model:
            experiment.log_parameters({
                "pretrained_model_path": args.pretrained_model,
                "pretrained_model_type": type(pretrained_model).__name__
            })
            if hasattr(pretrained_model, "get_params"):
                experiment.log_parameters(pretrained_model.get_params(deep=False))
                
        if args.cross_validate:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] if args.model != 'rfr' \
                     else ['neg_mean_squared_error', 'r2']
            
            model = train_model(
                X, y, args.sampling, args.model,
                tune=args.tune,
                recall_opt=args.recall_opt,
                tune_method=args.tune_method,
                early_stopping=args.early_stopping,
                stopping_rounds=args.stopping_rounds,
                pretrained_model=pretrained_model
            )
            
            cv_results = cross_validate(
                model, X, y, 
                cv=args.cv_folds, 
                scoring=scoring, 
                n_jobs=-1,
                return_train_score=True
            )
            
            logger.info(f"Результаты кросс-валидации:\n{json.dumps(cv_results, indent=4)}")
            experiment.log_metrics({
                f'cv_mean_{k}': np.mean(v) for k, v in cv_results.items()
                if not k.startswith('train_')
            })
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.15, 
                random_state=42, 
                stratify=y if args.model != 'rfr' else None
            )

            model = train_model(
                X_train, y_train, args.sampling, args.model,
                tune=args.tune,
                recall_opt=args.recall_opt,
                tune_method=args.tune_method,
                early_stopping=args.early_stopping,
                stopping_rounds=args.stopping_rounds,
                pretrained_model=pretrained_model
            )

            # Предсказания и оценка модели
            y_score = model.predict_proba(X_test)[:, 1] if args.model != 'rfr' else model.predict(X_test)

            user_threshold = args.threshold
            user_metrics = evaluate_model(y_test, y_score, user_threshold, f"{args.model}_{args.sampling}_user")

            optimal_threshold = find_optimal_threshold(y_test, y_score)
            optimal_metrics = evaluate_model(y_test, y_score, optimal_threshold, f"{args.model}_{args.sampling}_optimal")

            # Логирование метрик
            metrics_to_log = {}
            if args.model == 'rfr':
                metrics_to_log.update({
                    f"{args.model}_{args.sampling}_user_mse": user_metrics['mse'],
                    f"{args.model}_{args.sampling}_user_r2": user_metrics['r2'],
                    f"{args.model}_{args.sampling}_optimal_mse": optimal_metrics['mse'],
                    f"{args.model}_{args.sampling}_optimal_r2": optimal_metrics['r2']
                })
            else:
                metrics_to_log.update({
                    f"{args.model}_{args.sampling}_user_accuracy": user_metrics['accuracy'],
                    f"{args.model}_{args.sampling}_user_precision": user_metrics['precision'],
                    f"{args.model}_{args.sampling}_user_recall": user_metrics['recall'],
                    f"{args.model}_{args.sampling}_user_f1": user_metrics['f1'],
                    f"{args.model}_{args.sampling}_user_roc_auc": user_metrics['roc_auc'],
                    f"{args.model}_{args.sampling}_optimal_threshold": optimal_threshold,
                    f"{args.model}_{args.sampling}_optimal_f1": optimal_metrics['f1']
                })
            experiment.log_metrics(metrics_to_log)

            # Логирование дополнительных данных
            if args.model != 'rfr':
                y_pred_user = (y_score >= user_threshold).astype(int)
                experiment.log_confusion_matrix(y_test, y_pred_user, 
                                              title=f"Confusion Matrix (Threshold={user_threshold:.2f})")
                
                y_pred_optimal = (y_score >= optimal_threshold).astype(int)
                experiment.log_confusion_matrix(y_test, y_pred_optimal,
                                              title=f"Confusion Matrix (Optimal Threshold={optimal_threshold:.2f})")

            # Сохранение предсказаний
            X_full, _ = prepare_data(df)
            predictions_csv_path = save_predictions(
                df, model, X_full, 
                OUTPUT_DIR, 
                f"{args.model}_{args.sampling}", 
                threshold=user_threshold
            )
            experiment.log_asset(predictions_csv_path)

            # Визуализации
            try:
                plt_fig = plot_calibration_curve(
                    y_test, y_score, 
                    title=f"Calibration Curve ({args.model}, {args.sampling})"
                )
                experiment.log_figure(
                    figure=plt_fig, 
                    figure_name=f"calibration_curve_{args.model}_{args.sampling}.png"
                )
                plt_fig.close()
            except Exception as e:
                logger.error(f"Ошибка при создании calibration curve: {str(e)}")

            try:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                importance_plot_path = plot_feature_importance(
                    model, feature_names, 
                    title=f"{args.model}_{args.sampling}"
                )
                if importance_plot_path:
                    experiment.log_image(
                        importance_plot_path,
                        name=f"feature_importance_{args.model}_{args.sampling}"
                    )
            except Exception as e:
                logger.error(f"Ошибка при создании feature importance: {str(e)}")

        # Сохранение модели и отчета
        model_file = os.path.join(OUTPUT_DIR, f"model_{args.model}_{args.sampling}.joblib")
        joblib.dump(model, model_file)
        experiment.log_model(args.model, model_file)

        report_path = append_run_to_category_report(
            category_name=args.ctgr_name,
            start_time=start_time,
            model_type=args.model,
            user_metrics=user_metrics if not args.cross_validate else {},
            optimal_metrics=optimal_metrics if not args.cross_validate else {},
            user_threshold=user_threshold if not args.cross_validate else 0.5,
            optimal_threshold=optimal_threshold if not args.cross_validate else 0.5,
            sampling_method=args.sampling,
            comment=args.comment
        )
        if report_path:
            experiment.log_asset(report_path, file_name="classification_report.md")
            
        experiment.log_parameters({
            "output_dir": OUTPUT_DIR,
            "dataset_shape": str(X.shape) if 'X' in locals() else "None",
            "class_ratio": f"{sum(y)/len(y):.2f}" if 'y' in locals() and len(y) > 0 else "None",
            "optimal_threshold": optimal_threshold if 'optimal_threshold' in locals() else "None",
            "final_model_type": type(model).__name__ if 'model' in locals() else "None"
        })

    except Exception as e:
        logger.error(f"Ошибка выполнения: {str(e)}", exc_info=True)
        if 'experiment' in locals():
            experiment.log_parameters({
                "error": str(e),
                "status": "failed"
            })
            experiment.end()
        raise
    finally:
        if 'experiment' in locals():
            experiment.log_parameter("status", "completed")
            experiment.end()
        logger.info(f"Выполнение завершено. Результаты в {OUTPUT_DIR}")
