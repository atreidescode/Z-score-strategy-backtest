"""
data_loader.py — Chargement, validation et resampling des données OHLCV.
Absorbe la logique de 1m_to_1h-2.py et load_data() de l'ancien backtest.
"""

import logging
import os
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def load_h1(filepath: str) -> pd.DataFrame:
    """
    Charge un CSV H1 au format : Date_Time,O,H,L,C,V
    Lève FileNotFoundError si le fichier est absent.
    Lève ValueError si les colonnes requises sont manquantes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Date_Time"])
    df.rename(columns={"Date_Time": "DateTime"}, inplace=True)
    df.set_index("DateTime", inplace=True)

    _validate(df, filepath)

    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.sort_index(inplace=True)
    df = _clean(df)

    logger.info(
        "Chargé : %s | %d bougies | %s → %s",
        filepath, len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


def resample_m1_to_h1(files: List[str], output_path: str = "historique_3ans_H1.csv") -> pd.DataFrame:
    """
    Lit plusieurs fichiers CSV HistData (M1, séparateur ';'),
    les fusionne, resamples en H1 et sauvegarde le résultat.
    """
    frames: List[pd.DataFrame] = []

    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier M1 introuvable : {path}")
        logger.info("Lecture : %s", path)
        df_tmp = pd.read_csv(
            path, sep=";", header=None,
            names=["Date_Time", "Open", "High", "Low", "Close", "Volume"],
        )
        frames.append(df_tmp)

    df_all = pd.concat(frames, ignore_index=True)
    logger.info("Assemblage : %d bougies M1 au total", len(df_all))

    df_all["Date_Time"] = pd.to_datetime(df_all["Date_Time"], format="%Y%m%d %H%M%S")
    df_all.set_index("Date_Time", inplace=True)
    df_all.sort_index(inplace=True)

    df_h1 = df_all.resample("1h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()

    df_h1.index.name = "Date_Time"
    df_h1.to_csv(output_path)
    logger.info("Resampling terminé : %d bougies H1 → %s", len(df_h1), output_path)
    return df_h1


def _validate(df: pd.DataFrame, source: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source} — colonnes manquantes : {missing}")
    dupes = df.index.duplicated().sum()
    if dupes:
        logger.warning("%s — %d lignes dupliquées supprimées", source, dupes)
    nulls = df[list(REQUIRED_COLUMNS)].isnull().sum().sum()
    if nulls:
        logger.warning("%s — %d valeurs nulles détectées", source, nulls)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="first")]
    df.dropna(inplace=True)
    return df