import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import json
from typing import Dict, Any


def find_optimal_thresholds_fast(df):
    features = [col for col in df.columns if col != "label"]
    y_true = df["label"].values
    results = []
    for feature in features:
        x = df[feature].values
        # Нижний порог: feature > t
        precisions, recalls, thresholds = precision_recall_curve(y_true, x)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        # Синхронизируем длины
        min_len = min(len(f1s), len(thresholds))
        if min_len > 0:
            f1s_cut = f1s[:min_len]
            thresholds_cut = thresholds[:min_len]
            best_idx = np.nanargmax(f1s_cut)
            best_t_lower = thresholds_cut[best_idx]
            best_f1_lower = f1s_cut[best_idx]
        else:
            best_t_lower = None
            best_f1_lower = None
        # Верхний порог: feature < t
        precisions, recalls, thresholds = precision_recall_curve(y_true, -x)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        min_len = min(len(f1s), len(thresholds))
        if min_len > 0:
            f1s_cut = f1s[:min_len]
            thresholds_cut = thresholds[:min_len]
            best_idx = np.nanargmax(f1s_cut)
            best_t_upper = -thresholds_cut[best_idx]
            best_f1_upper = f1s_cut[best_idx]
        else:
            best_t_upper = None
            best_f1_upper = None
        results.append(
            {
                "feature": feature,
                "optimal_lower_threshold": best_t_lower,
                "f1_at_lower": best_f1_lower,
                "optimal_upper_threshold": best_t_upper,
                "f1_at_upper": best_f1_upper,
            }
        )
    return pd.DataFrame(results)


def _md_path(path):
    return path.replace("\\", "/")


def generate_markdown_report(
    results: Dict[str, Dict[str, Any]],
    output_folder: str,
    dataset_name: str,
    thresholds_df: pd.DataFrame,
) -> str:
    plots_dir = os.path.join(output_folder, "plots")
    reports_dir = os.path.join(output_folder, "reports_and_metrics")
    # Пути к картинкам
    hist_path = _md_path(
        os.path.relpath(os.path.join(plots_dir, "feature_histograms.png"), reports_dir)
    )
    roc_path = _md_path(
        os.path.relpath(
            os.path.join(plots_dir, "roc_curves_comparison.png"), reports_dir
        )
    )
    boxplot_path = _md_path(
        os.path.relpath(os.path.join(plots_dir, "feature_boxplots.png"), reports_dir)
    )
    # Сбор метрик моделей
    metrics_table = []
    for model_name, res in results.items():
        m = res["metrics"]
        metrics_table.append(
            [
                model_name,
                m.get("accuracy"),
                m.get("precision"),
                m.get("recall"),
                m.get("f1_score"),
                m.get("roc_auc_score"),
            ]
        )
    metrics_df = pd.DataFrame(
        metrics_table,
        columns=pd.Index(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]),
    )
    metrics_md = metrics_df.to_markdown(index=False, floatfmt=".3f")
    # Лучшая модель по ROC AUC
    best_row = metrics_df.sort_values("ROC AUC", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    # Пути к важности признаков и гиперпараметрам
    feature_importances_path = os.path.join(
        plots_dir, f"{best_model_name}_feature_importances.png"
    )
    feature_importances_md = _md_path(
        os.path.relpath(feature_importances_path, reports_dir)
    )
    if not os.path.exists(feature_importances_path):
        feature_importances_md = "*Важности признаков недоступны для этой модели*"
    best_params_path = os.path.join(
        output_folder, f"{best_model_name}_best_params.json"
    )
    if best_params_path and os.path.exists(best_params_path):
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
    else:
        best_params = {}
    # Таблица порогов
    thresholds_md = thresholds_df.to_markdown(index=False, floatfmt=".3f")
    # --- Формирование текста отчета ---
    roc_interp = f"*Интерпретация:* {best_model_name} показывает наилучшее качество, его кривая лежит выше остальных."
    report_md = f"""# Отчет по анализу для датасета: {dataset_name}

## 1. Распределение признаков
![Feature Histograms]({hist_path})
*Интерпретация:* Признаки с явным разделением классов могут быть полезны для простых правил или отбора признаков.

## 2. Сравнение моделей
### 2.1. Метрики качества
{metrics_md}

### 2.2. ROC-кривые
![ROC Curves]({roc_path})
{roc_interp}

## 3. Анализ лучшей модели: {best_model_name}
- **Лучшие гиперпараметры:**
```json
{json.dumps(best_params, ensure_ascii=False, indent=2)}
```
- **Важность признаков:**
{feature_importances_md}
*Интерпретация:* Наибольший вклад в решение вносят признаки с наибольшей важностью.

## 4. Пороговый анализ признаков
{thresholds_md}

*Интерпретация:* Некоторые признаки можно использовать как простые классификаторы с хорошим F1-score.
"""
    report_path = os.path.join(reports_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return report_path
