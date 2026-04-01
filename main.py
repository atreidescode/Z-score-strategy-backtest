"""
main.py — Point d'entrée unique de la stratégie SPX Mean Reversion.

Lancement :
    python main.py

Prérequis :
    pip install pandas numpy statsmodels matplotlib seaborn
"""

import logging
import sys
from dataclasses import replace

import pandas as pd

from config import Config
from data_loader import load_h1
from backtest import run_backtest
from optimizer import run_optimization
from plotting import (
    plot_page1_overview,
    plot_page2_performance,
    plot_page3_trades,
    plot_page4_optimization,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = Config()

    # 1. Chargement
    try:
        df = load_h1(cfg.data_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Astuce : lancez d'abord data_loader.resample_m1_to_h1() "
                    "pour générer le fichier H1 depuis les données M1 HistData.")
        sys.exit(1)

    # 2. Backtest théorique (sans frais)
    cfg_brut = replace(cfg, spread_points=0.0, swap_rate_annual=0.0, commission_per_trade=0.0)
    r_brut = run_backtest(df, cfg_brut, label="Théorique (sans frais)")

    # 3. Backtest simulation CFD
    r_cfd = run_backtest(df, cfg, label="Simulation CFD")

    # 4. Optimisation
    optim_df = None
    r_best = None
    if cfg.optimize:
        optim_df = run_optimization(df, cfg)
        if len(optim_df):
            best_row = optim_df.iloc[0]
            cfg_best = replace(cfg,
                               lookback_hours=int(best_row["lookback"]),
                               zscore_entry=float(best_row["entry_z"]),
                               zscore_exit=float(best_row["exit_z"]))
            r_best = run_backtest(df, cfg_best, label="Paramètres optimaux (CFD)")

    # 5. Visualisations
    plot_page1_overview(r_brut, r_cfd, cfg, r_best)
    plot_page2_performance(r_brut, r_cfd)
    plot_page3_trades(r_brut, r_cfd, cfg)
    plot_page4_optimization(optim_df, r_best)

    # 6. Export CSV
    r_brut["trades_df"].to_csv("trades_brut.csv", index=False)
    r_cfd["trades_df"].to_csv("trades_cfd.csv",   index=False)
    r_brut["equity_df"].to_csv("equity_brut.csv")
    r_cfd["equity_df"].to_csv("equity_cfd.csv")
    if optim_df is not None:
        optim_df.to_csv("optimization_results.csv", index=False)

    metric_keys = ["label", "total_return_pct", "cagr_pct", "sharpe", "calmar",
                   "max_drawdown_pct", "n_trades", "win_rate_pct", "profit_factor",
                   "avg_days_held", "expectancy", "final_capital"]
    rows = [{k: r[k] for k in metric_keys} for r in [r_brut, r_cfd] if r]
    if r_best:
        rows.append({k: r_best[k] for k in metric_keys})
    pd.DataFrame(rows).to_csv("metrics_summary.csv", index=False)

    logger.info("✅ Pipeline terminé.")


if __name__ == "__main__":
    main()