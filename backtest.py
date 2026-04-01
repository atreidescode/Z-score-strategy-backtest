"""
backtest.py — Moteur de backtest vectorisé pour la stratégie Mean Reversion SPX.

Amélioration clé : Z-scores et flags ADF précalculés en rolling vectorisé.
La boucle principale ne simule que l'exécution des ordres.
Logique métier strictement identique à l'original.
"""

import logging
import numpy as np
import pandas as pd

from config import Config
from indicators import rolling_zscore, rolling_adf_flag
from metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_backtest(df: pd.DataFrame, cfg: Config, label: str = "Backtest") -> dict:
    """
    Exécute le backtest mean reversion sur df avec les paramètres de cfg.
    """
    logger.info("── Backtest [%s] | Lookback=%dh | EntryZ=%.1f | ExitZ=%.1f",
                label, cfg.lookback_hours, cfg.zscore_entry, cfg.zscore_exit)

    close = df["Close"]
    returns_h = close.pct_change()

    # ── Précalcul vectorisé ───────────────────────────────────────────────────
    zscores = rolling_zscore(returns_h, cfg.lookback_hours)
    adf_flags = rolling_adf_flag(returns_h, cfg.lookback_hours, cfg.adf_pvalue)

    # ── Masque première bougie >= market_open_hour par jour ───────────────────
    daily_open_mask = _get_market_open_mask(returns_h.index, cfg.market_open_hour)
    signal_dates = returns_h.index[daily_open_mask]

    swap_daily = cfg.swap_rate_annual / 365.0
    capital = cfg.initial_capital
    position = entry_price = 0.0
    in_trade = False
    days_in_trade = 0

    equity_curve, trades, signals_log = [], [], []

    for ts in signal_dates:
        if ts not in zscores.index or pd.isna(zscores[ts]):
            continue

        current_price = close.loc[ts]
        z = float(zscores[ts])
        is_stationary = bool(adf_flags[ts]) if ts in adf_flags.index else False

        if in_trade and days_in_trade > 0:
            capital -= position * entry_price * swap_daily

        signals_log.append({
            "date": ts, "close": current_price,
            "z_score": z, "stationary": is_stationary, "in_trade": in_trade,
        })

        if is_stationary:
            if z < cfg.zscore_entry and not in_trade:
                buy_price = current_price + cfg.spread_points
                capital -= cfg.commission_per_trade
                position = capital / buy_price
                entry_price = buy_price
                in_trade = True
                days_in_trade = 0
                trades.append({
                    "entry_date": ts, "entry_price": buy_price, "entry_zscore": z,
                    "exit_date": None, "exit_price": None,
                    "pnl": None, "pnl_pct": None, "days_held": None,
                })

            elif z > cfg.zscore_exit and in_trade:
                sell_price = current_price - cfg.spread_points
                capital -= cfg.commission_per_trade
                pnl = position * (sell_price - entry_price)
                capital = position * sell_price
                trades[-1].update({
                    "exit_date": ts, "exit_price": sell_price,
                    "pnl": pnl,
                    "pnl_pct": (sell_price - entry_price) / entry_price * 100,
                    "days_held": days_in_trade,
                })
                in_trade = False
                position = 0.0
                days_in_trade = 0

        if in_trade:
            days_in_trade += 1

        equity_curve.append({
            "date": ts,
            "equity": (position * current_price) if in_trade else capital,
        })

    # ── Fermeture forcée du dernier trade ouvert ──────────────────────────────
    if in_trade and trades:
        last_price = close.iloc[-1] - cfg.spread_points
        trades[-1].update({
            "exit_date": close.index[-1], "exit_price": last_price,
            "pnl": position * (last_price - entry_price),
            "pnl_pct": (last_price - entry_price) / entry_price * 100,
            "days_held": days_in_trade,
        })

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades).dropna(subset=["exit_date"])
    signals_df = pd.DataFrame(signals_log).set_index("date")

    result = compute_metrics(equity_df, trades_df, cfg.initial_capital, label)
    result["signals_df"] = signals_df
    result["lookback_hours"] = cfg.lookback_hours
    return result


def _get_market_open_mask(index: pd.DatetimeIndex, open_hour: int) -> np.ndarray:
    """Retourne True pour la première bougie >= open_hour de chaque jour."""
    df_tmp = pd.DataFrame({"ts": index}, index=index)
    df_tmp["date"] = df_tmp["ts"].dt.normalize()
    df_tmp["hour"] = df_tmp["ts"].dt.hour
    df_tmp["valid"] = df_tmp["hour"] >= open_hour
    first_valid = df_tmp[df_tmp["valid"]].groupby("date")["ts"].first()
    return index.isin(first_valid.values)