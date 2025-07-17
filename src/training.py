import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import VotingClassifier
from keras import layers
from keras.callbacks import EarlyStopping
import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred  # fallback, если только классы
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc_score": roc_auc_score(y_test, y_proba),
    }
    return {"metrics": metrics, "probabilities": y_proba}


def train_sklearn_models(X_train, y_train, models_config, random_state) -> dict:
    results = {}
    for model_name, model_obj, param_grid in models_config:
        logging.info(f"Обучение {model_name}")
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model_obj)])
        param_grid_pipe = (
            {f"model__{k}": v for k, v in param_grid.items()} if param_grid else {}
        )
        grid_size = (
            np.prod([len(v) for v in param_grid_pipe.values()])
            if param_grid_pipe
            else 1
        )
        fit_params = {}
        # Для XGBClassifier динамически задаём scale_pos_weight через fit_params
        if model_name == "XGBClassifier":
            n_0 = np.sum(y_train == 0)
            n_1 = np.sum(y_train == 1)
            scale_pos_weight = n_0 / n_1 if n_1 > 0 else 1.0
            fit_params = {"model__scale_pos_weight": scale_pos_weight}
            logging.info(
                f"XGBClassifier: scale_pos_weight set to {scale_pos_weight:.3f}"
            )
        if grid_size <= 20:
            search = GridSearchCV(
                pipe, param_grid_pipe, scoring="roc_auc", cv=5, n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                pipe,
                param_grid_pipe,
                n_iter=min(25, int(grid_size)),
                scoring="roc_auc",
                cv=5,
                n_jobs=-1,
                random_state=random_state,
            )
        search.fit(X_train, y_train, **fit_params)
        best_model = search.best_estimator_
        best_params = search.best_params_
        results[model_name] = {"model": best_model, "best_params": best_params}
    return results


def train_keras_model(
    X_train, y_train, X_test, y_test, class_weight_dict
) -> keras.Model:
    input_dim = X_train.shape[1]
    model = keras.Sequential(
        [
            layers.Dense(32, activation="relu", input_shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose="auto",
        class_weight=class_weight_dict,
    )
    return model


def train_voting_ensemble(estimators: list, X_train, y_train) -> VotingClassifier:
    voting_clf = VotingClassifier(estimators=estimators, voting="soft")
    voting_clf.fit(X_train, y_train)
    return voting_clf
