import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def save_feature_histograms(df: pd.DataFrame, save_path: str) -> None:
    features = [col for col in df.columns if col != "label"]
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols
    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, ncols, i)
        for label, color, name in zip([0, 1], ["blue", "red"], ["No Smoke", "Smoke"]):
            plt.hist(
                df[df["label"] == label][feature],
                bins=30,
                alpha=0.6,
                label=name if i == 1 else None,
                color=color,
            )
        plt.title(feature)
        if i == 1:
            plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_feature_boxplots(df: pd.DataFrame, save_path: str) -> None:
    features = [col for col in df.columns if col != "label"]
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols
    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, ncols, i)
        sns.boxplot(
            data=df,
            x="label",
            y=feature,
            hue="label",
            palette=["blue", "red"],
            legend=False,
        )
        plt.title(feature)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_roc_curves(results: Dict, y_test: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(10, 8))
    for model_name, res in results.items():
        y_proba = res["probabilities"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_precision_recall_curve(
    results: Dict, y_test: np.ndarray, save_path: str
) -> None:
    plt.figure(figsize=(10, 8))
    for model_name, res in results.items():
        y_proba = res["probabilities"]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Comparison")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_per_feature_roc_curves(df: pd.DataFrame, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    features = [col for col in df.columns if col != "label"]
    for feature in features:
        plt.figure(figsize=(7, 5))
        y_true = df["label"]
        y_scores = df[feature]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {feature}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"roc_curve_{feature}.png")
        plt.savefig(save_path)
        plt.close()
