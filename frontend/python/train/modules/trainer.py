from __future__ import annotations

import threading
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from core.config import MODELS_CSV_DIR
from modules.csv_handler import get_uploaded_csv

training_states: dict[str, dict[str, Any]] = {}

ALGORITHMS = {
    "random_forest": {
        "label": "Random Forest",
        "description": "Ensemble of decision trees. Handles non-linear relationships, resistant to overfitting. Best default choice for classification and regression — works well without tuning.",
        "params": {
            "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 2000},
            "max_depth": {"type": "int", "default": 0, "min": 0, "max": 200},
        },
    },
    "dummy": {
        "label": "Dummy",
        "description": "Always predicts a constant value based on a simple strategy. Use as a reference to evaluate other models.",
        "params": {
            "strategy": {
                "type": "select",
                "default": "most_frequent",
                "options": ["most_frequent", "prior", "stratified", "uniform", "constant", "mean", "median", "quantile"],
            },
        },
    },
    "logistic_regression": {
        "label": "Logistic / Linear Regression",
        "description": "Simple, fast, interpretable. Logistic Regression for classification, Linear Regression for regression. Works best with linear relationships.",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
        },
    },
    "svm": {
        "label": "SVM",
        "description": "Support Vector Machine. Effective in high-dimensional spaces. Works well with clear margins of separation. Kernel trick handles non-linearity.",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
            "kernel": {
                "type": "select",
                "default": "rbf",
                "options": ["linear", "rbf", "poly", "sigmoid"],
            },
        },
    },
    "neural_network": {
        "label": "Neural Network (MLP)",
        "description": "Multi-layer Perceptron. Can learn complex non-linear patterns. Requires more data and hyperparameter tuning. Slower training.",
        "params": {
            "hidden_layers": {
                "type": "string",
                "default": "100",
                "hint": "comma-separated, e.g. 100,50",
            },
            "learning_rate": {
                "type": "float",
                "default": 0.001,
                "min": 0.00001,
                "max": 1.0,
            },
            "epochs": {"type": "int", "default": 1000, "min": 1, "max": 10000},
            "batch_size": {"type": "int", "default": 0, "min": 0, "max": 1024},
        },
    },
    "xgboost": {
        "label": "XGBoost",
        "description": "Gradient boosted trees. Often wins Kaggle competitions. Handles mixed data types, missing values. Can overfit on small datasets.",
        "params": {
            "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 2000},
            "max_depth": {"type": "int", "default": 6, "min": 1, "max": 50},
            "learning_rate": {
                "type": "float",
                "default": 0.3,
                "min": 0.001,
                "max": 1.0,
            },
        },
    },
}


def get_available_algorithms() -> dict[str, Any]:
    result = {}
    for key, cfg in ALGORITHMS.items():
        if key == "xgboost":
            try:
                import xgboost  # noqa: F401
            except ImportError:
                result[key] = dict(cfg, available=False)
                continue
        result[key] = dict(cfg, available=True)
    return result


def _build_model(
    algorithm: str, task: str, params: dict[str, Any]
) -> object:
    if algorithm == "dummy":
        strategy = str(params.get("strategy", ""))
        cls_strategies = {"most_frequent", "prior", "stratified", "uniform", "constant"}
        reg_strategies = {"mean", "median", "quantile", "constant"}
        if task == "classification":
            if strategy not in cls_strategies:
                strategy = "most_frequent"
            return DummyClassifier(strategy=strategy)
        else:
            if strategy not in reg_strategies:
                strategy = "mean"
            return DummyRegressor(strategy=strategy)

    if algorithm == "random_forest":
        md = params.get("max_depth", 0)
        kwargs = {
            "n_estimators": int(params.get("n_estimators", 100)),
            "random_state": 42,
        }
        if md and int(md) > 0:
            kwargs["max_depth"] = int(md)
        if task == "classification":
            return RandomForestClassifier(**kwargs)
        else:
            return RandomForestRegressor(**kwargs)

    elif algorithm == "logistic_regression":
        if task == "classification":
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=5000,
                random_state=42,
            )
        else:
            return LinearRegression()

    elif algorithm == "svm":
        kernel = params.get("kernel", "rbf")
        kwargs = {
            "C": float(params.get("C", 1.0)),
            "kernel": kernel,
        }
        if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
            kwargs["gamma"] = "scale"
        if task == "classification":
            return SVC(**kwargs, probability=True, random_state=42)
        else:
            return SVR(**kwargs)

    elif algorithm == "neural_network":
        hidden_raw = str(params.get("hidden_layers", "100"))
        try:
            hidden = tuple(int(x.strip()) for x in hidden_raw.split(",") if x.strip())
        except ValueError:
            hidden = (100,)
        kwargs = {
            "hidden_layer_sizes": hidden,
            "learning_rate_init": float(params.get("learning_rate", 0.001)),
            "max_iter": int(params.get("epochs", 200)),
            "batch_size": int(params.get("batch_size", 0)) or "auto",
            "random_state": 42,
        }
        if task == "classification":
            return MLPClassifier(**kwargs)
        else:
            return MLPRegressor(**kwargs)

    elif algorithm == "xgboost":
        import xgboost as xgb

        md = params.get("max_depth", 6)
        kwargs = {
            "n_estimators": int(params.get("n_estimators", 100)),
            "learning_rate": float(params.get("learning_rate", 0.3)),
            "random_state": 42,
            "verbosity": 0,
        }
        if md and int(md) > 0:
            kwargs["max_depth"] = int(md)
        if task == "classification":
            return xgb.XGBClassifier(**kwargs)
        else:
            return xgb.XGBRegressor(**kwargs)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _detect_task(y: pd.Series) -> str:
    unique = y.nunique()
    if y.dtype in ("int64", "int32", "int16", "int8", "bool", "uint8"):
        if unique <= 20:
            return "classification"
        return "regression"
    if y.dtype in ("float64", "float32", "float16"):
        return "regression"
    return "classification"


def _cancellable_fit(model: Any, X_train, y_train, cancelled_fn) -> bool:
    if isinstance(model, MLPClassifier):
        max_epochs = model.max_iter
        model.set_params(warm_start=True, max_iter=1)
        for _ in range(max_epochs):
            if cancelled_fn():
                return True
            model.fit(X_train, y_train)
        return False
    else:
        if cancelled_fn():
            return True
        model.fit(X_train, y_train)
        return False


def _train_worker(
    session_id: str,
    csv_path: str,
    target_col: str,
    feature_cols: list[str],
    algorithm: str,
    params: dict[str, Any],
    test_size: float,
    task_type: str = "auto",
    random_state: int = 42,
    shuffle: bool = True,
    val_size: float = 0.0,
    val_strategy: str = "simple",
    k_folds: int = 5,
) -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    state = training_states.setdefault(
        session_id,
        {
            "status": "idle",
            "log": [],
            "model_path": None,
            "metrics": None,
            "error": None,
        },
    )
    state["status"] = "running"
    state["log"] = []
    state["_cancel"] = False
    log = state["log"]

    MAX_LOG_LINES = 500

    def log_msg(msg: str) -> None:
        log.append(msg)
        if len(log) > MAX_LOG_LINES:
            log[: len(log) - MAX_LOG_LINES] = []

    try:
        if random_state == -1:
            random_state = None

        log_msg(f"Loading data from {Path(csv_path).name}...")
        df = pd.read_csv(csv_path)
        log_msg(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        y = df[target_col]
        X = df[feature_cols]

        if task_type == "auto":
            task = _detect_task(y)
            log_msg(f"Auto-detected task: {task}")
        else:
            task = task_type
            log_msg(f"Task: {task} (user-specified)")
        log_msg(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
        log_msg(f"Target: {target_col}")
        if task == "classification":
            class_dist = y.value_counts().to_dict()
            log_msg(f"Classes: {class_dist}")

        model = _build_model(algorithm, task, params)
        algo_label = ALGORITHMS.get(algorithm, {}).get("label", algorithm)
        metrics: dict[str, Any] = {"task": task, "algorithm": algo_label}
        y_test = None
        y_pred = None

        if val_strategy == "kfold":
            log_msg(f"K-fold cross-validation ({k_folds} folds)...")
            kf = KFold(n_splits=k_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
            fold_metrics: dict[str, list[float]] = {}
            all_y_test: list[float] = []
            all_y_pred: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                if state.get("_cancel"):
                    log_msg("Training cancelled by user")
                    state["status"] = "cancelled"
                    return
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                model_fold = _build_model(algorithm, task, params)
                if _cancellable_fit(model_fold, X_fold_train, y_fold_train, lambda: state.get("_cancel")):
                    log_msg("Training cancelled by user")
                    state["status"] = "cancelled"
                    return
                y_fold_pred = model_fold.predict(X_fold_val)

                all_y_test.extend(y_fold_val.tolist())
                all_y_pred.extend(y_fold_pred.tolist())

                if task == "classification":
                    fold_metrics.setdefault("accuracy", []).append(accuracy_score(y_fold_val, y_fold_pred))
                    fold_metrics.setdefault("precision", []).append(
                        precision_score(y_fold_val, y_fold_pred, average="weighted", zero_division=0)
                    )
                    fold_metrics.setdefault("recall", []).append(
                        recall_score(y_fold_val, y_fold_pred, average="weighted", zero_division=0)
                    )
                    fold_metrics.setdefault("f1_score", []).append(
                        f1_score(y_fold_val, y_fold_pred, average="weighted", zero_division=0)
                    )
                    try:
                        if hasattr(model_fold, "predict_proba"):
                            y_proba = model_fold.predict_proba(X_fold_val)
                            if y_proba.shape[1] == 2:
                                fold_metrics.setdefault("roc_auc", []).append(
                                    roc_auc_score(y_fold_val, y_proba[:, 1])
                                )
                    except Exception:
                        pass
                else:
                    fold_metrics.setdefault("mae", []).append(mean_absolute_error(y_fold_val, y_fold_pred))
                    fold_metrics.setdefault("mse", []).append(mean_squared_error(y_fold_val, y_fold_pred))
                    fold_metrics.setdefault("rmse", []).append(
                        float(np.sqrt(mean_squared_error(y_fold_val, y_fold_pred)))
                    )
                    fold_metrics.setdefault("r2", []).append(r2_score(y_fold_val, y_fold_pred))

                log_msg(f"  Fold {fold_idx + 1}/{k_folds} complete")

            y_test = all_y_test
            y_pred = all_y_pred

            for key, values in fold_metrics.items():
                metrics[key] = round(float(np.mean(values)), 4)
            log_msg(f"CV metrics (mean across folds): {metrics}")

            if state.get("_cancel"):
                log_msg("Training cancelled by user")
                state["status"] = "cancelled"
                return
            log_msg("Training final model on full dataset...")
            if _cancellable_fit(model, X, y, lambda: state.get("_cancel")):
                log_msg("Training cancelled by user")
                state["status"] = "cancelled"
                return
            log_msg("Final model trained on all data")

        else:
            log_msg(f"Shuffle: {shuffle}, test size: {test_size}, val size: {val_size}, seed: {random_state}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
            )
            log_msg(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

            if val_size > 0.0:
                val_ratio = val_size / (1.0 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=val_ratio, random_state=random_state, shuffle=shuffle
                )
                log_msg(f"Validation: {len(X_val)} samples")

            log_msg(f"Training {algorithm}...")

            fit_start = time.time()
            if _cancellable_fit(model, X_train, y_train, lambda: state.get("_cancel")):
                log_msg("Training cancelled by user")
                state["status"] = "cancelled"
                return
            fit_time = time.time() - fit_start
            log_msg(f"Training complete ({fit_time:.2f}s)")

            y_pred = model.predict(X_test)
            log_msg("Evaluating...")

            if task == "classification":
                metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
                metrics["precision"] = round(
                    precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
                )
                metrics["recall"] = round(
                    recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
                )
                metrics["f1_score"] = round(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
                )
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)
                        if y_proba.shape[1] == 2:
                            metrics["roc_auc"] = round(
                                roc_auc_score(y_test, y_proba[:, 1]), 4
                            )
                except Exception:
                    pass
            else:
                metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
                metrics["mse"] = round(mean_squared_error(y_test, y_pred), 4)
                metrics["rmse"] = round(
                    float(np.sqrt(mean_squared_error(y_test, y_pred))), 4
                )
                metrics["r2"] = round(r2_score(y_test, y_pred), 4)

            log_msg(f"Metrics: {metrics}")

        state["metrics"] = metrics
        state["_model"] = model
        state["_csv_path"] = csv_path
        state["_feature_cols"] = feature_cols
        state["_target_col"] = target_col

        y_test_list = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test) if y_test is not None else []
        y_pred_list = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred) if y_pred is not None else []
        state["_y_test"] = y_test_list
        state["_y_pred"] = y_pred_list

        tmp_dir = MODELS_CSV_DIR.parent / "temp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        model_file = tmp_dir / f"{Path(csv_path).stem}.model"
        joblib.dump(model, model_file)
        state["model_path"] = str(model_file)
        log_msg(f"Model saved to {model_file.name}")

        state["status"] = "done"

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        log_msg(f"ERROR: {e}")


def cancel_training(session_id: str) -> dict[str, Any]:
    state = training_states.get(session_id)
    if state:
        state["_cancel"] = True
    return {"status": "cancelling"}


def start_training(
    session_id: str,
    csv_path: str,
    target_col: str,
    feature_cols: list[str],
    algorithm: str,
    params: dict[str, Any],
    test_size: float = 0.2,
    task_type: str = "auto",
    random_state: int = 42,
    shuffle: bool = True,
    val_size: float = 0.0,
    val_strategy: str = "simple",
    k_folds: int = 5,
) -> dict[str, Any]:
    thread = threading.Thread(
        target=_train_worker,
        args=(
            session_id,
            csv_path,
            target_col,
            feature_cols,
            algorithm,
            params,
            test_size,
            task_type,
            random_state,
            shuffle,
            val_size,
            val_strategy,
            k_folds,
        ),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


def get_training_status(session_id: str) -> dict[str, Any]:
    state = training_states.get(session_id)
    if state is None:
        return {"status": "idle", "log": [], "metrics": None, "error": None}
    result: dict[str, Any] = {
        "status": state["status"],
        "log": list(state["log"]),
        "metrics": state["metrics"],
        "error": state["error"],
    }
    y_test = state.get("_y_test")
    y_pred = state.get("_y_pred")
    if y_test is not None and y_pred is not None:
        result["predictions"] = {"y_test": y_test, "y_pred": y_pred}
    return result


def save_to_server(session_id: str, name: str) -> str:
    state = training_states.get(session_id)
    if state is None or state["status"] != "done":
        raise ValueError("No trained model available")

    model = state.get("_model")
    if model is None:
        raise ValueError("Model reference lost")

    name = name.strip()
    if not name:
        csv_info = get_uploaded_csv(session_id)
        if csv_info and csv_info.get("filename"):
            name = Path(csv_info["filename"]).stem
        else:
            name = "model"

    if not name.endswith(".model"):
        name += ".model"

    models_dir = MODELS_CSV_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / name
    joblib.dump(model, dest)

    metrics = state.get("metrics") or {}
    metadata = dict(metrics)
    metadata["feature_cols"] = state.get("_feature_cols", [])
    metadata["target_col"] = state.get("_target_col", "")
    metadata["task"] = metrics.get("task", "auto")
    try:
        import json as _json
        index_path = models_dir / "_metrics.json"
        index: dict = {}
        if index_path.exists():
            with open(index_path) as f:
                index = _json.load(f)
        index[name] = metadata
        with open(index_path, "w") as f:
            _json.dump(index, f, indent=2)
    except Exception:
        pass

    return str(dest)
