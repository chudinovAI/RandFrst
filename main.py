import os
import logging
import gc
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
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

# fix logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Step 1 load data
raw_df = load_and_combine_data(
    ProjectConfig.SMOKE_DATA_PATH, ProjectConfig.NO_SMOKE_DATA_PATH
)
logging.info(
    f"Loaded {raw_df.shape[0]} lines, {raw_df.shape[1]} features (united DataFrame)"
)

# Step 2 creating dynamic features
features_df = create_dynamic_features(raw_df, window_size=5)
logging.info(
    f"After dynamic features: {features_df.shape[0]} lines, {features_df.shape[1]} features"
)


# Optimizing types and memory usage
def optimize_df_types(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logging.info(
        f"DataFrame memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB"
    )
    return df


# calling gc
logging.info("Memory: deleted raw_df and calling gc.collect()...")
del raw_df
gc.collect()

features_df = optimize_df_types(features_df)

# correlation analyze
corr = features_df.corr()
top_corr = corr[ProjectConfig.TARGET_COLUMN].abs().sort_values(ascending=False)[1:11]
logging.info("Top 10 features by correlation with label:")
for feat, val in top_corr.items():
    logging.info(f"  {feat}: {val:.3f}")
os.makedirs("outputs/plots", exist_ok=True)
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/plots/correlation_heatmap.png")
plt.close()

# sampling and trainig
features = [col for col in features_df.columns if col != ProjectConfig.TARGET_COLUMN]
X = features_df[features]
y = features_df[ProjectConfig.TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=ProjectConfig.TEST_SIZE,
    random_state=ProjectConfig.RANDOM_STATE,
    stratify=y,
)

# rus = RandomUnderSampler(random_state=ProjectConfig.RANDOM_STATE)
# logging.info("RandomUnderSampler...")
# X_train_res_np, y_train_res_np = rus.fit_resample(X_train, y_train)
# X_train_res = pd.DataFrame(X_train_res_np, columns=X_train.columns)
# y_train_res = pd.Series(y_train_res_np)
# logging.info(f"Size y_train before: {y_train.shape}, after: {y_train_res.shape}")

logging.info("SMOTEENN...")
sampler = SMOTEENN(random_state=ProjectConfig.RANDOM_STATE, n_jobs=-1)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
logging.info(f"Size y_train before: {y_train.shape}, after: {y_train_res.shape}")

# Keras's class_weight with y_train_res base
class_counts = Counter(y_train_res)
total = sum(class_counts.values())
class_weight = {cls: total / (2 * count) for cls, count in class_counts.items()}

# sklearn models learning
sklearn_results = train_sklearn_models(
    X_train_res,
    y_train_res,
    ProjectConfig.MODELS_AND_PARAMS,
    ProjectConfig.RANDOM_STATE,
)
# Keras learning
keras_model = train_keras_model(X_train_res, y_train_res, X_test, y_test, class_weight)

# analizing models collections
all_models = {name: res["model"] for name, res in sklearn_results.items()}
all_models = {"KerasNN": keras_model, **all_models}
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

# best 3 models by ROC(without KERAS)
top_models = sorted(
    ((name, res) for name, res in final_results.items() if name != "KerasNN"),
    key=lambda x: x[1]["metrics"]["roc_auc_score"],
    reverse=True,
)[:3]
estimators = [(name, res["model"]) for name, res in top_models]
if len(estimators) >= 2:
    voting_clf = train_voting_ensemble(estimators, X_train_res, y_train_res)
    eval_res = evaluate_model(voting_clf, X_test, y_test)
    final_results["VotingClassifier"] = {
        "model": voting_clf,
        "metrics": eval_res["metrics"],
        "probabilities": eval_res["probabilities"],
        "y_test": y_test,
        "best_params": None,
    }

# Reporting
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
