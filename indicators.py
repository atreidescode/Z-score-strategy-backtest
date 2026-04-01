"""
indicators.py — Fonctions pures de calcul d'indicateurs.
Aucun état global. Testables unitairement de façon isolée.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def test_stationarity(returns: pd.Series, pvalue_threshold: float = 0.05) -> bool:
    """Teste la stationnarité via ADF. Retourne True si stationnaire."""
    try:
        return adfuller(returns.dropna())[1] < pvalue_threshold
    except Exception:
        return False


def get_zscore(returns: pd.Series) -> float:
    """Z-score de la dernière valeur. Retourne 0.0 si std == 0."""
    std = returns.std()
    if std == 0:
        return 0.0
    return float((returns.iloc[-1] - returns.mean()) / std)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Z-score glissant vectorisé sur toute la série."""
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)


def rolling_adf_flag(series: pd.Series, window: int, pvalue_threshold: float = 0.05) -> pd.Series:
    """
    ADF glissant vectorisé via rolling.apply.
    Retourne une série booléenne : True = stationnaire sur cette fenêtre.
    """
    def _adf_pvalue(arr: np.ndarray) -> float:
        try:
            return adfuller(arr)[1]
        except Exception:
            return 1.0

    pvalues = series.rolling(window).apply(_adf_pvalue, raw=True)
    return pvalues < pvalue_threshold