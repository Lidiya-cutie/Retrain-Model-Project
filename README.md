# Retrain Model Project

## Описание проекта

Этот проект представляет собой пайплайн для повторного обучения моделей машинного обучения на уже подготовленных датасетах. Код поддерживает различные алгоритмы (RandomForest, XGBoost, LightGBM), стратегии балансировки классов, настройку гиперпараметров и логирование экспериментов.

## Основные возможности

- Поддержка нескольких типов моделей:
  - RandomForestClassifier
  - XGBClassifier
  - LGBMClassifier
  - RandomForestRegressor (для задач регрессии)
  
- Методы балансировки классов:
  - SMOTE
  - BorderlineSMOTE
  - ADASYN
  - RandomOverSampler
  - RandomUnderSampler
  - Без балансировки

- Возможности тюнинга гиперпараметров:
  - Random Search
  - Grid Search
  - Optuna (байесовская оптимизация)
  
- Интеграция с Comet.ml для логирования экспериментов
- Поддержка кросс-валидации
- Генерация отчетов и визуализаций
- Возможность дообучения предварительно обученных моделей

## Требования

Для работы проекта необходимы следующие зависимости:

```
numpy
pandas
scikit-learn
imbalanced-learn
xgboost
lightgbm
optuna
comet-ml
matplotlib
joblib
```

## Использование

### Основные параметры запуска

```bash
python retrain_model.py \
    --model rf \             # Тип модели (rf, xgb, lgb, rfr)
    --sampling smote \       # Метод балансировки (none, smote, borderline и др.)
    --threshold 0.5 \       # Порог классификации
    --tune \                # Включить тюнинг гиперпараметров
    --tune-method optuna \  # Метод тюнинга (random_search, optuna, grid_search)
    --pretrained-model path/to/model.joblib \  # Путь к предобученной модели
    --early-stopping \      # Включить раннюю остановку
    --cross-validate \      # Использовать кросс-валидацию
    --ctgr-name "Category Name" \  # Имя категории для отчетов
    --comment "Test run"    # Комментарий к прогону
```

### Примеры команд

1. Обучение RandomForest с SMOTE и тюнингом:
```bash
python retrain_model.py --model rf --sampling smote --tune
```

2. Обучение XGBoost с оптимизацией под recall:
```bash
python retrain_model.py --model xgb --recall-opt --tune --tune-method optuna
```

3. Дообучение существующей модели:
```bash
python retrain_model.py --model xgb --pretrained-model old_model.joblib
```

## Выходные данные

Скрипт создает следующие артефакты:

- Обученная модель (формат .joblib)
- CSV файл с предсказаниями
- JSON файлы с метриками качества
- Графики важности признаков
- Калибровочные кривые
- Отчет в формате Markdown
- Логи экспериментов в Comet.ml

## Структура проекта

```
scripts/                 # Скрипты для запуска
retrain_model.py          # Основной скрипт
hyperparam_tuning.py            # Модуль для тюнинга (опционально)
secrets.json             # Файл с API ключами (не включен в репозиторий)
retrained_results/       # Директория с результатами
reports/                 # Генерируемые отчеты
images/                  # Графики
```

## Настройка Comet.ml

Для логирования экспериментов необходимо:
1. Создать аккаунт на comet.ml
2. Добавить API ключ в secrets.json
3. Указать workspace в коде (по умолчанию "lidiya-cutie")

## Особенности реализации

- Поддержка как классификации, так и регрессии
- Автоматическое определение необходимости балансировки
- Интеллектуальная обработка ошибок
- Гибкая система логирования и отчетности
- Поддержка различных стратегий оптимизации

## Лицензия

Проект распространяется под лицензией MIT.
