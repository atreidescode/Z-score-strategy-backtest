"""
optimizer.py — Grid search sur les paramètres de la stratégie.
Teste toutes les combinaisons (lookback × zscore_entry × zscore_exit)
et retourne un DataFrame trié par Sharpe ratio décroissant.
"""

import logging
from itertools import product
from dataclasses import replace

import pandas as pd

from config import Config
from backtest import run_backtest

logger = logging.getLogger(__name__)


def run_optimization(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Grid search sur les plages définies dans cfg."""
    combos = [
        (lb, ze, zx)
        for lb, ze, zx in product(
            cfg.lookback_range,
            cfg.zscore_entry_range,
            cfg.zscore_exit_range,
        )
        if ze < zx
    ]

    logger.info("[OPTIM] Grid search : %d combinaisons à tester", len(combos))
    results = []

    for i, (lb, ze, zx) in enumerate(combos, 1):
        cfg_run = replace(cfg, lookback_hours=lb, zscore_entry=ze, zscore_exit=zx)
        try:
            m = run_backtest(df, cfg_run, label=f"LB={lb}|Ze={ze}|Zx={zx}")
            results.append({
                "lookback": lb, "entry_z": ze, "exit_z": zx,
                "sharpe": m["sharpe"], "cagr": m["cagr_pct"],
                "max_dd": m["max_drawdown_pct"], "win_rate": m["win_rate_pct"],
                "n_trades": m["n_trades"], "profit_factor": m["profit_factor"],
                "total_return": m["total_return_pct"],
            })
        except Exception as e:
            logger.warning("[SKIP] LB=%d Ze=%.1f Zx=%.1f — %s", lb, ze, zx, e)

        if i % 5 == 0:
            logger.info("  ... %d/%d combinaisons testées", i, len(combos))

    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    logger.info("[OPTIM] Terminé. Top 5 :\n%s", df_res.head(5).to_string(index=False))
    return df_res