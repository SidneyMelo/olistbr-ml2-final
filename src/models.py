"""
Módulo de modelos supervisionados.

Implementa funções para:
- Divisão treino/teste
- Treino de Naive Bayes
- Treino de Regressão (Linear/Ridge)
- Treino de SVM
- Treino de RandomForest e XGBoost (opcional)
- Avaliação (classification_report, RMSE, etc.)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def _stabilize_nb(X):
    # Clip para evitar overflow/underflow em NB
    return np.clip(np.nan_to_num(X, copy=False, posinf=0, neginf=0), -15, 15)


def _stabilize_logreg(X):
    # Clip para evitar overflow/underflow no SGDClassifier
    return np.clip(np.nan_to_num(X, copy=False, posinf=0, neginf=0), -20, 20)


@dataclass
class TrainTestData:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> TrainTestData:
    """
    Divide os dados em treino e teste.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
    random_state : int
    stratify : bool

    Returns
    -------
    TrainTestData
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return TrainTestData(X_train, X_test, y_train, y_test)


# =========================
#   CLASSIFICAÇÃO
# =========================

def train_naive_bayes_classifier(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    var_smoothing: float = 1e-4,
    variance_threshold: float = 1e-4,
) -> Pipeline:
    """
    Treina um classificador Naive Bayes (Gaussian) em pipeline com preprocessor.

    Parameters
    ----------
    preprocessor : ColumnTransformer
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    Pipeline
    """
    clf = GaussianNB(var_smoothing=var_smoothing)
    stabilizer = FunctionTransformer(
        _stabilize_nb,
        feature_names_out="one-to-one",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("stabilizer", stabilizer),
            ("var_thresh", VarianceThreshold(threshold=variance_threshold)),
            ("clf", clf),
        ]
    )
    model.fit(X_train, y_train)
    # Evita divisao por zero em atributos com variancia zero por classe
    try:
        nb = model.named_steps["clf"]
        if hasattr(nb, "sigma_"):
            nb.sigma_ = np.maximum(nb.sigma_, var_smoothing)
        if hasattr(nb, "var_"):
            nb.var_ = np.maximum(nb.var_, var_smoothing)
    except Exception:
        # se atributos nao existirem, segue sem ajuste
        pass
    return model


def train_logistic_regression_classifier(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 0.5,
    max_iter: int = 2000,
    class_weight: Optional[Union[Dict, str]] = None,
) -> Pipeline:
    """
    Treina um classificador de Regressão Logística (para comparação com NB e SVM).

    Parameters
    ----------
    preprocessor : ColumnTransformer
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    Pipeline
    """
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.001,
        max_iter=max_iter,
        tol=1e-3,
        learning_rate="adaptive",
        eta0=0.01,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
    )

    stabilizer = FunctionTransformer(
        _stabilize_logreg,
        feature_names_out="one-to-one",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("stabilizer", stabilizer),
            ("clf", clf),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_svm_classifier(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: Union[str, float] = "scale",
    class_weight: Optional[Union[Dict, str]] = None,
) -> Pipeline:
    """
    Treina um classificador SVM em pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
    X_train : pd.DataFrame
    y_train : pd.Series
    kernel : str
    C : float
    gamma : str or float

    Returns
    -------
    Pipeline
    """
    clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_classifier(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Pipeline:
    """
    Treina um RandomForestClassifier.

    Parameters
    ----------
    preprocessor : ColumnTransformer
    X_train : pd.DataFrame
    y_train : pd.Series
    n_estimators : int
    random_state : int

    Returns
    -------
    Pipeline
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_classifier(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Optional[Pipeline]:
    """
    Treina um XGBoostClassifier, se a biblioteca estiver instalada.

    Returns
    -------
    Optional[Pipeline]
        Retorna None se xgboost não estiver disponível.
    """
    if not HAS_XGBOOST:
        print("XGBoost não está instalado. Pulando esse modelo.")
        return None

    clf = XGBClassifier(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Avalia um modelo de classificação: accuracy, matriz de confusão,
    classification_report.

    Parameters
    ----------
    model : Pipeline
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


# =========================
#   REGRESSÃO
# =========================

def train_linear_regression(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ridge: bool = False,
    alpha: float = 1.0,
) -> Pipeline:
    """
    Treina um modelo de Regressão Linear ou Ridge (com regularização).

    Parameters
    ----------
    preprocessor : ColumnTransformer
    X_train : pd.DataFrame
    y_train : pd.Series
    ridge : bool
        Se True, usa Ridge Regression. Caso contrário, LinearRegression.
    alpha : float
        Parâmetro de regularização do Ridge.

    Returns
    -------
    Pipeline
    """
    if ridge:
        reg = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.001,
            learning_rate="adaptive",
            eta0=0.01,
            max_iter=2000,
            tol=1e-3,
            random_state=42,
        )
    else:
        reg = SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="adaptive",
            eta0=0.01,
            max_iter=2000,
            tol=1e-3,
            random_state=42,
        )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("reg", reg),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Avalia um modelo de regressão usando MAE, RMSE e R².

    Parameters
    ----------
    model : Pipeline
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
    """
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # R²
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
