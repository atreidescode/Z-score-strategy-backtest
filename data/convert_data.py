"""
convert_data.py — Conversion des données brutes M1 → H1.
Format source : 20240101 180000;Open;High;Low;Close;Volume (sans header, séparateur ;)

Lancement :
    python convert_data.py
"""

import os
import logging
import pandas as pd
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HERE = os.path.abspath(os.path.dirname(__file__))

FICHIERS_M1: List[str] = [
    os.path.join(HERE, "DAT_ASCII_SPXUSD_M1_2023.csv"),
    os.path.join(HERE, "DAT_ASCII_SPXUSD_M1_2024.csv"),
    os.path.join(HERE, "DAT_ASCII_SPXUSD_M1_2025.csv"),
]

OUTPUT: str = os.path.join(HERE, "historique_3ans_H1.csv")


def convert_m1_to_h1(files: List[str], output_path: str) -> pd.DataFrame:
    frames = []

    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        logger.info("Lecture : %s", path)

        df_tmp = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["Date_Time", "Open", "High", "Low", "Close", "Volume"],
        )

        # Format exact : "20240101 180000"
        df_tmp["Date_Time"] = pd.to_datetime(
            df_tmp["Date_Time"],
            format="%Y%m%d %H%M%S",
        )

        df_tmp.set_index("Date_Time", inplace=True)
        df_tmp = df_tmp.astype(float)
        logger.info("  → %d bougies M1 chargées", len(df_tmp))
        frames.append(df_tmp)

    logger.info("Assemblage...")
    df_all = pd.concat(frames)
    df_all.sort_index(inplace=True)

    dupes = df_all.index.duplicated().sum()
    if dupes:
        logger.warning("%d doublons supprimés", dupes)
        df_all = df_all[~df_all.index.duplicated(keep="first")]

    logger.info("Resampling M1 → H1...")
    df_h1 = df_all.resample("1h").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()

    df_h1.index.name = "Date_Time"
    df_h1.to_csv(output_path)
    logger.info("Fichier sauvegardé : %s", output_path)
    return df_h1


if __name__ == "__main__":
    print(f"Dossier : {HERE}")
    print(f"Output  : {OUTPUT}\n")

    df = convert_m1_to_h1(files=FICHIERS_M1, output_path=OUTPUT)

    print("\n" + "─" * 45)
    print("  ✅ CONVERSION TERMINÉE")
    print("─" * 45)
    print(f"  Bougies H1 générées : {len(df):,}")
    print(f"  Période             : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Fichier généré      : {OUTPUT}")
    print("─" * 45)
    print("\nAperçu :")
    print(df.head())
    print(f"\n➡️  Vous pouvez maintenant lancer : python spx_strategy/main.py")