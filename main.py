import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import ProjectConfig
from src.data_processing import load_and_combine_data, create_dynamic_features
from src.training import (
    train_sklearn_models,
    train_keras_model,
    train_voting_ensemble,
    evaluate_model,
)
from src.visualization import (
    save_feature_histograms,
    save_feature_boxplots,
    save_roc_curves,
    save_per_feature_roc_curves,
    save_precision_recall_curve,
)
from src.reporting import generate_markdown_report, find_optimal_thresholds_fast
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Шаг 1: Загрузка и сборка данных ===
raw_df = load_and_combine_data(
    ProjectConfig.SMOKE_DATA_PATH, ProjectConfig.NO_SMOKE_DATA_PATH
)
logging.info(f"Загружено {raw_df.shape[0]} строк, {raw_df.shape[1]} признаков")

# === Шаг 2: Создание динамических признаков ===
features_df = create_dynamic_features(raw_df, window_size=5)
logging.info(
    f"После динамических признаков: {features_df.shape[0]} строк, {features_df.shape[1]} признаков"
)

# === Шаг 3: Анализ корреляции ===
corr = features_df.corr()
top_corr = corr[ProjectConfig.TARGET_COLUMN].abs().sort_values(ascending=False)[1:11]
logging.info("Топ-10 признаков по корреляции с label:")
for feat, val in top_corr.items():
    logging.info(f"  {feat}: {val:.3f}")
# Сохраняем heatmap
os.makedirs("outputs/plots", exist_ok=True)
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/plots/correlation_heatmap.png")
plt.close()

# === Шаг 4: Разделение данных и обучение ===
features = [col for col in features_df.columns if col != ProjectConfig.TARGET_COLUMN]
X = features_df[features]
y = features_df[ProjectConfig.TARGET_COLUMN]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=ProjectConfig.TEST_SIZE,
    random_state=ProjectConfig.RANDOM_STATE,
    stratify=y,
)
# class_weight для Keras
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weight = {cls: total / (2 * count) for cls, count in class_counts.items()}
# Обучение sklearn моделей
sklearn_results = train_sklearn_models(
    X_train, y_train, ProjectConfig.MODELS_AND_PARAMS, ProjectConfig.RANDOM_STATE
)
# Обучение Keras
keras_model = train_keras_model(X_train, y_train, X_test, y_test, class_weight)
# Собираем все модели
all_models = {name: res["model"] for name, res in sklearn_results.items()}
all_models = {"KerasNN": keras_model, **all_models}
# Оценка моделей
final_results = {}
for name, model in all_models.items():
    eval_res = evaluate_model(model, X_test, y_test)
    best_params = (
        sklearn_results[name]["best_params"] if name in sklearn_results else None
    )
    final_results[name] = {
        "model": model,
        "metrics": eval_res["metrics"],
        "probabilities": eval_res["probabilities"],
        "y_test": y_test,
        "best_params": best_params,
    }
# Ансамбль
top_models = sorted(
    ((name, res) for name, res in final_results.items() if name != "KerasNN"),
    key=lambda x: x[1]["metrics"]["roc_auc_score"],
    reverse=True,
)[:3]
estimators = [(name, res["model"]) for name, res in top_models]
if len(estimators) >= 2:
    voting_clf = train_voting_ensemble(estimators, X_train, y_train)
    eval_res = evaluate_model(voting_clf, X_test, y_test)
    final_results["VotingClassifier"] = {
        "model": voting_clf,
        "metrics": eval_res["metrics"],
        "probabilities": eval_res["probabilities"],
        "y_test": y_test,
        "best_params": None,
    }

# === Шаг 5: Визуализация и отчетность ===
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports_and_metrics", exist_ok=True)
save_roc_curves(final_results, y_test, "outputs/plots/roc_curves_comparison.png")
save_precision_recall_curve(
    final_results, y_test, "outputs/plots/pr_curves_comparison.png"
)
save_feature_histograms(features_df, "outputs/plots/feature_histograms.png")
save_feature_boxplots(features_df, "outputs/plots/feature_boxplots.png")
save_per_feature_roc_curves(features_df, "outputs/plots/per_feature_roc_curves")
thresholds_df = find_optimal_thresholds_fast(features_df)
thresholds_df.to_csv("outputs/reports_and_metrics/feature_thresholds.csv", index=False)
generate_markdown_report(final_results, "outputs", "experiment", thresholds_df)
