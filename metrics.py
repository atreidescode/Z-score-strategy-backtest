"""
metrics.py — Calcul des métriques de performance d'un backtest.
Fonction pure : reçoit equity + trades, retourne un dict de métriques.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    label: str = "Backtest",
) -> dict:
    equity = equity_df["equity"]

    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.01)
    cagr = ((equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1) * 100

    running_max = equity.cummax()
    max_dd = ((equity - running_max) / running_max * 100).min()

    daily = equity.resample("1D").last().pct_change().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

    n_trades = len(trades_df)
    win_rate = avg_win = avg_loss = profit_factor = avg_days_held = expectancy = 0.0

    if n_trades > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / n_trades * 100
        avg_win = wins["pnl_pct"].mean() if len(wins) else 0.0
        avg_loss = losses["pnl_pct"].mean() if len(losses) else 0.0
        pf_denom = losses["pnl"].sum()
        profit_factor = abs(wins["pnl"].sum() / pf_denom) if pf_denom != 0 else np.inf
        avg_days_held = trades_df["days_held"].mean()
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    metrics = {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(profit_factor, 2),
        "avg_days_held": round(avg_days_held, 1),
        "expectancy": round(expectancy, 3),
        "final_capital": round(equity.iloc[-1], 2),
        "equity_df": equity_df,
        "trades_df": trades_df,
        "initial_capital": initial_capital,
    }

    sign = "+" if total_return >= 0 else ""
    logger.info(
        "[%s] Return=%s%.2f%% | CAGR=%+.2f%% | Sharpe=%.2f | MaxDD=%.2f%% | Trades=%d | WinRate=%.1f%% | PF=%.2f",
        label, sign, total_return, cagr, sharpe, max_dd, n_trades, win_rate, profit_factor,
    )
    return metrics