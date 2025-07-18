from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier


class ProjectConfig:
    # Пути к основным датасетам
    # SMOKE_DATA_PATH = "data/smoke_data.csv"
    # NO_SMOKE_DATA_PATH = "data/smoke_data_no_label_1.csv"
    SMOKE_DATA_PATH = "/kaggle/input/dataset/data/smoke_data.csv"
    NO_SMOKE_DATA_PATH = "/kaggle/input/dataset/data/smoke_data_no_label_1.csv"
    # Целевая переменная
    TARGET_COLUMN = "label"

    # Модели и параметры (имя, объект_модели, сетка_параметров)
    MODELS_AND_PARAMS = [
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=1000, solver="liblinear", class_weight="balanced"
            ),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            },
        ),
        (
            "AdaBoostClassifier",
            AdaBoostClassifier(random_state=42),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
        ),
        (
            "XGBClassifier",
            XGBClassifier(eval_metric="logloss", random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "eval_metric": ["logloss"],
            },
        ),
    ]
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
