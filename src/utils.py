"""
Utilitários gerais do projeto.

Inclui:
- Função para fixar semente aleatória (reprodutibilidade)
- Funções para salvar/carregar modelos (joblib)
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import random


def set_global_seed(seed: int = 42) -> None:
    """
    Define seed global para random, numpy (e futuramente torch, etc.).

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch é opcional
        pass


def save_model(model: Any, path: str | Path) -> None:
    """
    Salva um modelo treinado em disco usando joblib.

    Parameters
    ----------
    model : Any
        Modelo treinado (sklearn, xgboost, etc.).
    path : str | Path
        Caminho para o arquivo .pkl ou .joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    """
    Carrega um modelo salvo em disco via joblib.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Any
    """
    path = Path(path)
    return joblib.load(path)
