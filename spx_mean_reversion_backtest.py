"""
=============================================================================
SPX/USD — Mean Reversion Backtest
Stratégie : ADF + Z-Score sur returns horaires (adapté de QuantConnect)
Données   : historique_3ans_H1.csv  (format : Date_Time,O,H,L,C,V)
=============================================================================

INSTALLATION :
  pip install pandas numpy statsmodels matplotlib seaborn

LANCEMENT :
  python spx_mean_reversion_backtest.py

SORTIES :
  - PAGE 1 : spx_page1_overview.png        → Vue d'ensemble + métriques clés
  - PAGE 2 : spx_page2_performance.png     → Equity curves détaillées
  - PAGE 3 : spx_page3_trades.png          → Analyse des trades
  - PAGE 4 : spx_page4_optimization.png    → Résultats grid search
  - trades_brut.csv / trades_cfd.csv
  - equity_brut.csv / equity_cfd.csv
  - optimization_results.csv
  - metrics_summary.csv
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings
import os
from itertools import product
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# =============================================================================
# PALETTE & STYLE GLOBAUX
# =============================================================================

BG        = '#0B0E1A'   # fond principal
BG2       = '#131729'   # fond cartes
PANEL     = '#1A1F35'   # fond panneaux
GREEN     = '#00E5A0'   # gains / théorique
RED       = '#FF4C6E'   # pertes / CFD
YELLOW    = '#FFD166'   # meilleur paramètre
BLUE      = '#4EA8DE'   # z-scores / neutre
GREY      = '#4A5568'   # textes secondaires
WHITE     = '#E8EDF5'   # textes principaux
ACCENT    = '#7C5CFC'   # accent violet

def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    BG2,
        'axes.edgecolor':    PANEL,
        'axes.labelcolor':   WHITE,
        'xtick.color':       GREY,
        'ytick.color':       GREY,
        'text.color':        WHITE,
        'grid.color':        PANEL,
        'grid.linewidth':    0.6,
        'legend.facecolor':  PANEL,
        'legend.edgecolor':  GREY,
        'font.family':       'monospace',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG = {
    "data_path":            "historique_3ans_H1.csv",
    "lookback_hours":       500,
    "zscore_entry":        -1.0,
    "zscore_exit":          1.0,
    "adf_pvalue":           0.05,
    "market_open_hour":     9,
    "market_open_minute":   30,
    "initial_capital":  1_000_000,
    "top_n_combinations":     5,      # Nombre de meilleures combinaisons à afficher
    "spread_points":        0.5,
    "swap_rate_annual":     0.05,
    "commission_per_trade": 0.0,
    "optimize":             True,
    "lookback_range":       [300, 400, 500, 600],
    "zscore_entry_range":   [-0.5, -1.0, -1.5, -2.0],
    "zscore_exit_range":    [0.5,  1.0,  1.5],
}

# =============================================================================
# 2. CHARGEMENT
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    print(f"[DATA] Chargement : {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\n❌ Fichier introuvable : {filepath}")
    df = pd.read_csv(filepath, parse_dates=['Date_Time'])
    df.rename(columns={'Date_Time': 'DateTime'}, inplace=True)
    df.set_index('DateTime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    print(f"[DATA] {len(df):,} bougies | {df.index[0].date()} → {df.index[-1].date()}")
    return df

# =============================================================================
# 3. FONCTIONS CŒUR
# =============================================================================

def test_stationarity(returns, pvalue_threshold=0.05):
    try:
        return adfuller(returns.dropna())[1] < pvalue_threshold
    except Exception:
        return False

def get_zscore(returns):
    std = returns.std()
    return 0.0 if std == 0 else (returns.iloc[-1] - returns.mean()) / std

# =============================================================================
# 4. BACKTEST
# =============================================================================

def run_backtest(df, lookback_hours=500, zscore_entry=-1.0, zscore_exit=1.0,
                 adf_pvalue=0.05, initial_capital=100_000, spread_points=0.0,
                 swap_rate_annual=0.0, commission=0.0,
                 market_open_hour=9, market_open_minute=30, label="Backtest"):

    print(f"\n{'─'*55}")
    print(f"  ▶  {label}")
    print(f"     Lookback={lookback_hours}h | Entry Z={zscore_entry} | Exit Z={zscore_exit}")
    print(f"     Spread={spread_points}pts | Swap={swap_rate_annual*100:.1f}%/an")
    print(f"{'─'*55}")

    close     = df['Close']
    returns_h = close.pct_change().dropna()
    days      = returns_h.index.normalize().unique()
    swap_d    = swap_rate_annual / 365

    capital = initial_capital
    position = entry_price = 0.0
    in_trade = False
    days_in_trade = 0

    equity_curve, trades, signals_log = [], [], []

    for day in days:
        day_bars = returns_h[returns_h.index.normalize() == day]
        if day_bars.empty:
            continue
        open_bars   = day_bars[day_bars.index.hour >= market_open_hour]
        signal_time = open_bars.index[0] if not open_bars.empty else day_bars.index[0]

        idx = returns_h.index.get_loc(signal_time)
        if idx < lookback_hours:
            continue

        window = returns_h.iloc[idx - lookback_hours: idx]

        if in_trade and days_in_trade > 0:
            capital -= position * entry_price * swap_d

        is_stationary = test_stationarity(window, pvalue_threshold=adf_pvalue)
        z_score       = get_zscore(window)
        current_price = close.loc[signal_time]

        signals_log.append({
            'date': signal_time, 'close': current_price,
            'z_score': z_score, 'stationary': is_stationary, 'in_trade': in_trade,
        })

        if is_stationary:
            if z_score < zscore_entry and not in_trade:
                buy_price   = current_price + spread_points
                capital    -= commission
                position    = capital / buy_price
                entry_price = buy_price
                in_trade    = True
                days_in_trade = 0
                trades.append({
                    'entry_date': signal_time, 'entry_price': buy_price,
                    'entry_zscore': z_score, 'exit_date': None,
                    'exit_price': None, 'pnl': None, 'pnl_pct': None, 'days_held': None,
                })
            elif z_score > zscore_exit and in_trade:
                sell_price = current_price - spread_points
                capital   -= commission
                pnl        = position * (sell_price - entry_price)
                capital    = position * sell_price
                trades[-1].update({
                    'exit_date': signal_time, 'exit_price': sell_price,
                    'pnl': pnl,
                    'pnl_pct': (sell_price - entry_price) / entry_price * 100,
                    'days_held': days_in_trade,
                })
                in_trade = False
                position = 0.0
                days_in_trade = 0

        if in_trade:
            days_in_trade += 1

        equity_curve.append({
            'date': signal_time,
            'equity': (position * current_price) if in_trade else capital
        })

    if in_trade and trades:
        last_price = close.iloc[-1] - spread_points
        trades[-1].update({
            'exit_date': close.index[-1], 'exit_price': last_price,
            'pnl': position * (last_price - entry_price),
            'pnl_pct': (last_price - entry_price) / entry_price * 100,
            'days_held': days_in_trade,
        })

    equity_df  = pd.DataFrame(equity_curve).set_index('date')
    trades_df  = pd.DataFrame(trades).dropna(subset=['exit_date'])
    signals_df = pd.DataFrame(signals_log).set_index('date')

    metrics = _compute_metrics(equity_df, trades_df, initial_capital, label)
    metrics.update({
        'equity_df': equity_df, 'trades_df': trades_df,
        'signals_df': signals_df, 'initial_capital': initial_capital,
        'lookback_hours': lookback_hours,
    })
    return metrics

# =============================================================================
# 5. MÉTRIQUES
# =============================================================================

def _compute_metrics(equity_df, trades_df, initial_capital, label):
    equity       = equity_df['equity']
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    n_years      = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr         = ((equity.iloc[-1] / initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100
    running_max  = equity.cummax()
    max_dd       = ((equity - running_max) / running_max * 100).min()
    daily        = equity.resample('1D').last().pct_change().dropna()
    sharpe       = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0
    calmar       = abs(cagr / max_dd) if max_dd != 0 else 0

    n_trades = len(trades_df)
    if n_trades > 0:
        wins          = trades_df[trades_df['pnl'] > 0]
        losses        = trades_df[trades_df['pnl'] <= 0]
        win_rate      = len(wins) / n_trades * 100
        avg_win       = wins['pnl_pct'].mean()   if len(wins)   else 0
        avg_loss      = losses['pnl_pct'].mean() if len(losses) else 0
        pf_d          = losses['pnl'].sum()
        profit_factor = abs(wins['pnl'].sum() / pf_d) if pf_d != 0 else np.inf
        avg_days_held = trades_df['days_held'].mean()
        expectancy    = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_days_held = expectancy = 0

    m = {
        'label': label,
        'total_return_pct': round(total_return, 2),
        'cagr_pct':         round(cagr, 2),
        'sharpe':           round(sharpe, 2),
        'calmar':           round(calmar, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'n_trades':         n_trades,
        'win_rate_pct':     round(win_rate, 2),
        'avg_win_pct':      round(avg_win, 3),
        'avg_loss_pct':     round(avg_loss, 3),
        'profit_factor':    round(profit_factor, 2),
        'avg_days_held':    round(avg_days_held, 1),
        'expectancy':       round(expectancy, 3),
        'final_capital':    round(equity.iloc[-1], 2),
    }

    sign = '+' if total_return >= 0 else ''
    print(f"\n  📊 {label}")
    print(f"  ┌────────────────────────────────────┐")
    print(f"  │ Return total  : {sign}{total_return:.2f}%")
    print(f"  │ CAGR          : {'+' if cagr>=0 else ''}{cagr:.2f}%/an")
    print(f"  │ Sharpe        : {sharpe:.2f}")
    print(f"  │ Max Drawdown  : {max_dd:.2f}%")
    print(f"  │ Nb trades     : {n_trades}")
    print(f"  │ Win rate      : {win_rate:.1f}%")
    print(f"  │ Profit Factor : {profit_factor:.2f}")
    print(f"  │ Capital final : ${equity.iloc[-1]:,.0f}")
    print(f"  └────────────────────────────────────┘")
    return m

# =============================================================================
# 6. OPTIMISATION
# =============================================================================

def run_optimization(df, config):
    combos = [(lb, ze, zx)
              for lb, ze, zx in product(config['lookback_range'],
                                        config['zscore_entry_range'],
                                        config['zscore_exit_range'])
              if ze < zx]

    print(f"\n[OPTIM] Grid search : {len(combos)} combinaisons")
    results = []
    for i, (lb, ze, zx) in enumerate(combos):
        try:
            m = run_backtest(df, lookback_hours=lb, zscore_entry=ze, zscore_exit=zx,
                             adf_pvalue=config['adf_pvalue'],
                             initial_capital=config['initial_capital'],
                             spread_points=config['spread_points'],
                             swap_rate_annual=config['swap_rate_annual'],
                             market_open_hour=config['market_open_hour'],
                             label=f"LB={lb}|Ze={ze}|Zx={zx}")
            results.append({'lookback': lb, 'entry_z': ze, 'exit_z': zx,
                             'sharpe': m['sharpe'], 'cagr': m['cagr_pct'],
                             'max_dd': m['max_drawdown_pct'], 'win_rate': m['win_rate_pct'],
                             'n_trades': m['n_trades'], 'profit_factor': m['profit_factor'],
                             'total_return': m['total_return_pct']})
        except Exception as e:
            print(f"  [SKIP] LB={lb} Ze={ze} Zx={zx} — {e}")
        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}/{len(combos)}")

    df_res = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(f"\n[OPTIM] TOP 10 :\n{df_res.head(10).to_string(index=False)}")
    return df_res

# =============================================================================
# 7. UTILITAIRES GRAPHIQUES
# =============================================================================

def _section_title(ax, text, color=WHITE):
    """Titre de section dans un axe."""
    ax.set_title(text, fontsize=11, fontweight='bold', color=color,
                 pad=10, loc='left')

def _badge(ax, x, y, text, color, fontsize=8.5):
    """Badge coloré dans un axe (coordonnées axes 0-1)."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', color=BG,
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color, edgecolor='none'),
            va='center', ha='left')

def _metric_card(fig, rect, title, value, subtitle='', color=GREEN, fontsize_val=22):
    """Carte de métrique (fig-level en coordonnées figure)."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    ax.text(0.5, 0.72, title, ha='center', va='center',
            fontsize=7.5, color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.42, value, ha='center', va='center',
            fontsize=fontsize_val, fontweight='bold', color=color, transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.14, subtitle, ha='center', va='center',
                fontsize=7, color=GREY, transform=ax.transAxes)
    return ax

# =============================================================================
# 8. PAGE 1 — VUE D'ENSEMBLE
# =============================================================================

def plot_page1_overview(rb, rc, r_best=None):
    apply_dark_style()
    fig = plt.figure(figsize=(20, 13), facecolor=BG)

    # ── Header ──────────────────────────────────────────────────────────────
    header_ax = fig.add_axes([0, 0.91, 1, 0.09])
    header_ax.set_facecolor(BG)
    header_ax.axis('off')
    header_ax.text(0.5, 0.72, 'SPX/USD — MEAN REVERSION STRATEGY',
                   ha='center', va='center', fontsize=20, fontweight='bold',
                   color=WHITE, transform=header_ax.transAxes)
    header_ax.text(0.5, 0.22,
                   'ADF Stationarity Filter  ·  Z-Score Signal  ·  Long Only  ·  3 ans de données',
                   ha='center', va='center', fontsize=10, color=GREY,
                   transform=header_ax.transAxes)
    # Ligne de séparation
    header_ax.axhline(0.02, color=ACCENT, linewidth=1.5, alpha=0.6)

    # ── Labels des scénarios ─────────────────────────────────────────────
    label_ax = fig.add_axes([0, 0.87, 1, 0.04])
    label_ax.axis('off')
    label_ax.set_facecolor(BG)
    for x, col, txt in [
        (0.18, GREEN, '● THÉORIQUE — Sans frais, potentiel brut maximum'),
        (0.55, RED,   '● SIMULATION CFD — Spread + financement overnight'),
    ]:
        label_ax.text(x, 0.5, txt, ha='left', va='center',
                      fontsize=9, color=col, fontweight='bold',
                      transform=label_ax.transAxes)
    if r_best:
        label_ax.text(0.82, 0.5, '★ PARAMÈTRES OPTIMAUX', ha='left', va='center',
                      fontsize=9, color=YELLOW, fontweight='bold',
                      transform=label_ax.transAxes)

    # ── Cartes métriques THÉORIQUE (ligne haute) ─────────────────────────
    metrics_brut = [
        ('RETURN TOTAL',   f"{rb['total_return_pct']:+.1f}%",   '3 ans'),
        ('CAGR',           f"{rb['cagr_pct']:+.1f}%",           'par an'),
        ('SHARPE',         f"{rb['sharpe']:.2f}",               'risk-adjusted'),
        ('MAX DRAWDOWN',   f"{rb['max_drawdown_pct']:.1f}%",    'pire période'),
        ('WIN RATE',       f"{rb['win_rate_pct']:.0f}%",        f"{rb['n_trades']} trades"),
        ('PROFIT FACTOR',  f"{rb['profit_factor']:.2f}",        'wins/losses'),
    ]
    for i, (title, val, sub) in enumerate(metrics_brut):
        x = 0.01 + i * 0.165
        col = RED if ('DRAWDOWN' in title or 'LOSS' in title.upper()) else GREEN
        _metric_card(fig, [x, 0.72, 0.155, 0.14], title, val, sub, color=col)

    # ── Cartes métriques CFD (ligne basse) ───────────────────────────────
    metrics_cfd = [
        ('RETURN TOTAL',   f"{rc['total_return_pct']:+.1f}%",   '3 ans'),
        ('CAGR',           f"{rc['cagr_pct']:+.1f}%",           'par an'),
        ('SHARPE',         f"{rc['sharpe']:.2f}",               'risk-adjusted'),
        ('MAX DRAWDOWN',   f"{rc['max_drawdown_pct']:.1f}%",    'pire période'),
        ('WIN RATE',       f"{rc['win_rate_pct']:.0f}%",        f"{rc['n_trades']} trades"),
        ('PROFIT FACTOR',  f"{rc['profit_factor']:.2f}",        'wins/losses'),
    ]
    for i, (title, val, sub) in enumerate(metrics_cfd):
        x = 0.01 + i * 0.165
        col = RED if ('DRAWDOWN' in title or 'LOSS' in title.upper()) else RED
        _metric_card(fig, [x, 0.56, 0.155, 0.14], title, val, sub, color=RED)

    # ── Equity curve comparative ──────────────────────────────────────────
    ax_eq = fig.add_axes([0.01, 0.28, 0.63, 0.26])
    ax_eq.set_facecolor(BG2)
    _section_title(ax_eq, '📈  PERFORMANCE CUMULÉE — Les 3 scénarios comparés')

    eq_b = rb['equity_df']['equity'] / rb['initial_capital'] * 100 - 100
    eq_c = rc['equity_df']['equity'] / rc['initial_capital'] * 100 - 100

    ax_eq.plot(eq_b.index, eq_b.values, color=GREEN, lw=2,
               label='Théorique (sans frais)', zorder=3)
    ax_eq.plot(eq_c.index, eq_c.values, color=RED,   lw=2,
               label='Simulation CFD',         zorder=3)
    ax_eq.fill_between(eq_b.index, eq_b.values, 0,
                       where=(eq_b.values >= 0), alpha=0.08, color=GREEN)
    ax_eq.fill_between(eq_b.index, eq_b.values, 0,
                       where=(eq_b.values < 0),  alpha=0.08, color=RED)
    ax_eq.axhline(0, color=GREY, lw=0.8, ls='--', alpha=0.5)

    if r_best:
        eq_best = r_best['equity_df']['equity'] / r_best['initial_capital'] * 100 - 100
        ax_eq.plot(eq_best.index, eq_best.values, color=YELLOW, lw=1.5,
                   ls='--', label='Paramètres optimaux (CFD)', zorder=4)

    ax_eq.set_ylabel('Return (%)', color=GREY, fontsize=9)
    ax_eq.legend(fontsize=8, loc='upper left')
    ax_eq.grid(True, alpha=0.15)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_eq.tick_params(labelsize=8)

    # ── Drawdown côte à côte ──────────────────────────────────────────────
    ax_dd = fig.add_axes([0.66, 0.28, 0.33, 0.26])
    ax_dd.set_facecolor(BG2)
    _section_title(ax_dd, '📉  DRAWDOWN')

    for res, color, lbl in [(rb, GREEN, 'Théorique'), (rc, RED, 'CFD')]:
        eq = res['equity_df']['equity']
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.45, color=color, label=lbl)
        ax_dd.plot(dd.index, dd.values, color=color, lw=0.8, alpha=0.7)

    ax_dd.set_ylabel('Drawdown (%)', color=GREY, fontsize=9)
    ax_dd.legend(fontsize=8)
    ax_dd.grid(True, alpha=0.15)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_dd.tick_params(labelsize=8)

    # ── Tableau comparatif ────────────────────────────────────────────────
    ax_tab = fig.add_axes([0.01, 0.03, 0.97, 0.23])
    ax_tab.axis('off')
    ax_tab.set_facecolor(BG)

    ax_tab.text(0.0, 0.97, '📋  TABLEAU COMPARATIF COMPLET',
                fontsize=10, fontweight='bold', color=WHITE,
                transform=ax_tab.transAxes, va='top')

    rows = [
        ('Return Total (3 ans)',
         f"{rb['total_return_pct']:+.2f}%",
         f"{rc['total_return_pct']:+.2f}%",
         f"{r_best['total_return_pct']:+.2f}%" if r_best else '—'),
        ('CAGR (annualisé)',
         f"{rb['cagr_pct']:+.2f}%",
         f"{rc['cagr_pct']:+.2f}%",
         f"{r_best['cagr_pct']:+.2f}%" if r_best else '—'),
        ('Ratio de Sharpe',
         f"{rb['sharpe']:.2f}",
         f"{rc['sharpe']:.2f}",
         f"{r_best['sharpe']:.2f}" if r_best else '—'),
        ('Ratio de Calmar',
         f"{rb['calmar']:.2f}",
         f"{rc['calmar']:.2f}",
         f"{r_best['calmar']:.2f}" if r_best else '—'),
        ('Max Drawdown',
         f"{rb['max_drawdown_pct']:.2f}%",
         f"{rc['max_drawdown_pct']:.2f}%",
         f"{r_best['max_drawdown_pct']:.2f}%" if r_best else '—'),
        ('Nombre de trades',
         f"{rb['n_trades']}",
         f"{rc['n_trades']}",
         f"{r_best['n_trades']}" if r_best else '—'),
        ('Win Rate',
         f"{rb['win_rate_pct']:.1f}%",
         f"{rc['win_rate_pct']:.1f}%",
         f"{r_best['win_rate_pct']:.1f}%" if r_best else '—'),
        ('Profit Factor',
         f"{rb['profit_factor']:.2f}",
         f"{rc['profit_factor']:.2f}",
         f"{r_best['profit_factor']:.2f}" if r_best else '—'),
        ('Avg Win',
         f"{rb['avg_win_pct']:.3f}%",
         f"{rc['avg_win_pct']:.3f}%",
         f"{r_best['avg_win_pct']:.3f}%" if r_best else '—'),
        ('Avg Loss',
         f"{rb['avg_loss_pct']:.3f}%",
         f"{rc['avg_loss_pct']:.3f}%",
         f"{r_best['avg_loss_pct']:.3f}%" if r_best else '—'),
        ('Capital final ($1M départ)',
         f"${rb['final_capital']:>10,.0f}",
         f"${rc['final_capital']:>10,.0f}",
         f"${r_best['final_capital']:>10,.0f}" if r_best else '—'),
    ]

    col_x   = [0.01, 0.32, 0.55, 0.78]
    headers = ['MÉTRIQUE', '🟢 THÉORIQUE (sans frais)', '🔴 SIMULATION CFD', '⭐ PARAMÈTRES OPTIMAUX']
    h_colors = [WHITE, GREEN, RED, YELLOW]

    for j, (hdr, hcol) in enumerate(zip(headers, h_colors)):
        ax_tab.text(col_x[j], 0.87, hdr, fontsize=8.5, fontweight='bold',
                    color=hcol, transform=ax_tab.transAxes, va='top')

    ax_tab.plot([0, 1], [0.83, 0.83], color=GREY, lw=0.5, transform=ax_tab.transAxes, clip_on=False)

    for i, row in enumerate(rows):
        y = 0.78 - i * 0.065
        bg_color = PANEL if i % 2 == 0 else BG2
        rect = mpatches.FancyBboxPatch((0, y - 0.025), 1, 0.058,
                                        boxstyle="round,pad=0.005",
                                        facecolor=bg_color, edgecolor='none',
                                        transform=ax_tab.transAxes, clip_on=False)
        ax_tab.add_patch(rect)
        for j, val in enumerate(row):
            col = WHITE if j == 0 else (GREEN if j == 1 else (RED if j == 2 else YELLOW))
            ax_tab.text(col_x[j], y, val, fontsize=8, color=col,
                        transform=ax_tab.transAxes, va='center', fontfamily='monospace')

    plt.savefig('spx_page1_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
    print('[OUTPUT] ✅  spx_page1_overview.png')
    plt.close()

# =============================================================================
# 9. PAGE 2 — PERFORMANCE DÉTAILLÉE
# =============================================================================

def plot_page2_performance(rb, rc):
    apply_dark_style()
    fig = plt.figure(figsize=(20, 14), facecolor=BG)

    # Header
    hax = fig.add_axes([0, 0.95, 1, 0.05])
    hax.axis('off')
    hax.set_facecolor(BG)
    hax.text(0.5, 0.6, 'PAGE 2 — ANALYSE DE PERFORMANCE DÉTAILLÉE',
             ha='center', fontsize=15, fontweight='bold', color=WHITE)
    hax.axhline(0.05, color=ACCENT, lw=1.5, alpha=0.6)

    scenarios = [
        (rb, GREEN, '🟢 THÉORIQUE — Sans aucun frais (potentiel brut maximum)',
         'Ce que la stratégie ferait dans un monde idéal sans coûts de transaction'),
        (rc, RED,   '🔴 SIMULATION CFD — Avec spread 0.5pt + financement overnight 5%/an',
         'Ce que tu obtiendrais réellement en tradant ce signal sur un CFD SPX'),
    ]

    for idx_s, (res, color, title, subtitle) in enumerate(scenarios):
        top = 0.68 - idx_s * 0.38

        # Titre du scénario
        sax = fig.add_axes([0, top + 0.09, 1, 0.04])
        sax.axis('off')
        sax.set_facecolor(BG)
        sax.text(0.01, 0.5, title, fontsize=11, fontweight='bold', color=color)
        sax.text(0.01, 0.05, subtitle, fontsize=8.5, color=GREY)

        eq = res['equity_df']['equity']

        # Equity
        ax1 = fig.add_axes([0.01, top, 0.44, 0.085])
        ax1.set_facecolor(BG2)
        pct = eq / res['initial_capital'] * 100 - 100
        ax1.plot(pct.index, pct.values, color=color, lw=1.5)
        ax1.fill_between(pct.index, pct.values, 0,
                         where=(pct.values >= 0), alpha=0.1, color=color)
        ax1.fill_between(pct.index, pct.values, 0,
                         where=(pct.values < 0),  alpha=0.1, color=RED)
        ax1.axhline(0, color=GREY, lw=0.7, ls='--', alpha=0.5)
        ax1.set_title('Equity curve (% return)', fontsize=9, color=GREY, loc='left')
        ax1.set_ylabel('%', color=GREY, fontsize=8)
        ax1.grid(True, alpha=0.12)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.tick_params(labelsize=7)

        # Drawdown
        ax2 = fig.add_axes([0.47, top, 0.27, 0.085])
        ax2.set_facecolor(BG2)
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.6, color=RED)
        ax2.plot(dd.index, dd.values, color=RED, lw=0.8)
        ax2.set_title('Drawdown (%)', fontsize=9, color=GREY, loc='left')
        ax2.set_ylabel('%', color=GREY, fontsize=8)
        ax2.grid(True, alpha=0.12)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.tick_params(labelsize=7)

        # Returns mensuels
        ax3 = fig.add_axes([0.76, top, 0.23, 0.085])
        ax3.set_facecolor(BG2)
        monthly = eq.resample('ME').last().pct_change().dropna() * 100
        bar_colors = [GREEN if v >= 0 else RED for v in monthly.values]
        ax3.bar(range(len(monthly)), monthly.values, color=bar_colors, alpha=0.8, width=0.7)
        ax3.axhline(0, color=GREY, lw=0.7)
        ax3.set_title('Returns mensuels (%)', fontsize=9, color=GREY, loc='left')
        ax3.set_ylabel('%', color=GREY, fontsize=8)
        ax3.set_xticks(range(0, len(monthly), 3))
        ax3.set_xticklabels(
            [monthly.index[i].strftime('%m/%y') for i in range(0, len(monthly), 3)],
            fontsize=6, rotation=45
        )
        ax3.grid(True, alpha=0.12, axis='y')
        ax3.tick_params(labelsize=7)

    # Returns annuels comparatifs
    ax_yr = fig.add_axes([0.01, 0.04, 0.55, 0.15])
    ax_yr.set_facecolor(BG2)
    ax_yr.set_title('📅  RETURNS ANNUELS COMPARÉS — Théorique vs CFD',
                    fontsize=9, color=WHITE, loc='left', pad=8)

    for res, color, lbl in [(rb, GREEN, 'Théorique'), (rc, RED, 'CFD')]:
        eq = res['equity_df']['equity']
        ann = eq.resample('YE').last()
        yr  = ann.pct_change().dropna() * 100
        yrs = [str(d.year) for d in yr.index]
        x   = np.arange(len(yrs))
        offset = -0.2 if color == GREEN else 0.2
        bars = ax_yr.bar(x + offset, yr.values, width=0.38,
                         color=color, alpha=0.8, label=lbl)
        for bar, val in zip(bars, yr.values):
            ax_yr.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + (0.3 if val >= 0 else -1.5),
                       f"{val:+.1f}%", ha='center', fontsize=7.5,
                       color=color, fontweight='bold')

    ax_yr.set_xticks(np.arange(len(yrs)))
    ax_yr.set_xticklabels(yrs, fontsize=9)
    ax_yr.axhline(0, color=GREY, lw=0.7)
    ax_yr.set_ylabel('Return (%)', color=GREY, fontsize=8)
    ax_yr.legend(fontsize=8)
    ax_yr.grid(True, alpha=0.12, axis='y')

    # Rolling Sharpe
    ax_rs = fig.add_axes([0.59, 0.04, 0.40, 0.15])
    ax_rs.set_facecolor(BG2)
    ax_rs.set_title('📊  ROLLING SHARPE (90 jours) — Stabilité dans le temps',
                    fontsize=9, color=WHITE, loc='left', pad=8)

    for res, color, lbl in [(rb, GREEN, 'Théorique'), (rc, RED, 'CFD')]:
        eq = res['equity_df']['equity']
        daily = eq.resample('1D').last().pct_change().dropna()
        roll_sharpe = daily.rolling(90).mean() / daily.rolling(90).std() * np.sqrt(252)
        ax_rs.plot(roll_sharpe.index, roll_sharpe.values, color=color, lw=1.2, label=lbl)

    ax_rs.axhline(0, color=GREY, lw=0.7, ls='--')
    ax_rs.axhline(1, color=YELLOW, lw=0.7, ls=':', alpha=0.6, label='Sharpe=1')
    ax_rs.set_ylabel('Sharpe', color=GREY, fontsize=8)
    ax_rs.legend(fontsize=8)
    ax_rs.grid(True, alpha=0.12)
    ax_rs.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_rs.tick_params(labelsize=7)

    plt.savefig('spx_page2_performance.png', dpi=150, bbox_inches='tight', facecolor=BG)
    print('[OUTPUT] ✅  spx_page2_performance.png')
    plt.close()

# =============================================================================
# 10. PAGE 3 — ANALYSE DES TRADES
# =============================================================================

def plot_page3_trades(rb, rc):
    apply_dark_style()
    fig = plt.figure(figsize=(20, 13), facecolor=BG)

    hax = fig.add_axes([0, 0.95, 1, 0.05])
    hax.axis('off')
    hax.text(0.5, 0.6, 'PAGE 3 — ANALYSE MICROSCOPIQUE DES TRADES',
             ha='center', fontsize=15, fontweight='bold', color=WHITE)
    hax.axhline(0.05, color=ACCENT, lw=1.5, alpha=0.6)

    # ── Scénario label
    lax = fig.add_axes([0, 0.91, 1, 0.04])
    lax.axis('off')
    for x, col, txt in [
        (0.01, GREEN, '🟢 THÉORIQUE'),
        (0.17, RED,   '🔴 SIMULATION CFD'),
    ]:
        lax.text(x, 0.5, txt, fontsize=9, color=col, fontweight='bold',
                 transform=lax.transAxes)

    tb = rb['trades_df']
    tc = rc['trades_df']

    # ── 1. Distribution PnL
    ax1 = fig.add_axes([0.01, 0.63, 0.30, 0.25])
    ax1.set_facecolor(BG2)
    _section_title(ax1, 'Distribution PnL par trade (%)')
    for t, col, lbl in [(tb, GREEN, 'Théorique'), (tc, RED, 'CFD')]:
        if len(t):
            ax1.hist(t['pnl_pct'].dropna(), bins=25, alpha=0.6,
                     color=col, label=lbl, edgecolor='none')
    ax1.axvline(0, color=WHITE, lw=1.2, ls='--', alpha=0.7)
    ax1.set_xlabel('PnL (%)', fontsize=8, color=GREY)
    ax1.set_ylabel('Fréquence', fontsize=8, color=GREY)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.12)

    # ── 2. Durée des trades
    ax2 = fig.add_axes([0.34, 0.63, 0.30, 0.25])
    ax2.set_facecolor(BG2)
    _section_title(ax2, 'Distribution durée des trades (jours)')
    for t, col, lbl in [(tb, GREEN, 'Théorique'), (tc, RED, 'CFD')]:
        if len(t) and 'days_held' in t.columns:
            ax2.hist(t['days_held'].dropna(), bins=20, alpha=0.6,
                     color=col, label=lbl, edgecolor='none')
    ax2.set_xlabel('Jours en position', fontsize=8, color=GREY)
    ax2.set_ylabel('Fréquence', fontsize=8, color=GREY)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.12)

    # ── 3. PnL cumulatif trade par trade
    ax3 = fig.add_axes([0.67, 0.63, 0.32, 0.25])
    ax3.set_facecolor(BG2)
    _section_title(ax3, 'PnL cumulatif trade par trade')
    for t, col, lbl in [(tb, GREEN, 'Théorique'), (tc, RED, 'CFD')]:
        if len(t):
            cumul = t['pnl'].dropna().cumsum()
            ax3.plot(range(len(cumul)), cumul.values, color=col, lw=1.5, label=lbl)
            ax3.fill_between(range(len(cumul)), cumul.values, 0,
                             where=(cumul.values >= 0), alpha=0.08, color=col)
    ax3.axhline(0, color=GREY, lw=0.7, ls='--')
    ax3.set_xlabel('Numéro du trade', fontsize=8, color=GREY)
    ax3.set_ylabel('PnL cumulé ($)', fontsize=8, color=GREY)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.12)

    # ── 4. Z-score d'entrée vs PnL
    ax4 = fig.add_axes([0.01, 0.34, 0.30, 0.25])
    ax4.set_facecolor(BG2)
    _section_title(ax4, "Z-score d'entrée vs PnL (%)")
    for t, col, lbl in [(tb, GREEN, 'Théorique'), (tc, RED, 'CFD')]:
        if len(t):
            ax4.scatter(t['entry_zscore'], t['pnl_pct'],
                        color=col, alpha=0.5, s=18, label=lbl, edgecolors='none')
    ax4.axhline(0, color=GREY, lw=0.7, ls='--')
    ax4.axvline(CONFIG['zscore_entry'], color=YELLOW, lw=0.8, ls=':', alpha=0.8)
    ax4.set_xlabel("Z-score d'entrée", fontsize=8, color=GREY)
    ax4.set_ylabel('PnL (%)', fontsize=8, color=GREY)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.12)

    # ── 5. Durée vs PnL
    ax5 = fig.add_axes([0.34, 0.34, 0.30, 0.25])
    ax5.set_facecolor(BG2)
    _section_title(ax5, 'Durée (jours) vs PnL (%)')
    for t, col, lbl in [(tb, GREEN, 'Théorique'), (tc, RED, 'CFD')]:
        if len(t) and 'days_held' in t.columns:
            ax5.scatter(t['days_held'], t['pnl_pct'],
                        color=col, alpha=0.5, s=18, label=lbl, edgecolors='none')
    ax5.axhline(0, color=GREY, lw=0.7, ls='--')
    ax5.set_xlabel('Jours en position', fontsize=8, color=GREY)
    ax5.set_ylabel('PnL (%)', fontsize=8, color=GREY)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.12)

    # ── 6. Wins vs Losses par mois
    ax6 = fig.add_axes([0.67, 0.34, 0.32, 0.25])
    ax6.set_facecolor(BG2)
    _section_title(ax6, 'Trades gagnants vs perdants par mois')
    for t, col_w, col_l, lbl in [
        (tb, GREEN, '#1a6640', 'Théo'),
        (tc, '#ff8fa3', RED,   'CFD')
    ]:
        if len(t):
            t2 = t.copy()
            t2['month'] = pd.to_datetime(t2['entry_date']).dt.to_period('M')
            grp = t2.groupby('month')['pnl'].apply(
                lambda x: pd.Series({'wins': (x > 0).sum(), 'losses': (x <= 0).sum()})
            ).unstack()
            months = [str(m) for m in grp.index]
            x = np.arange(len(months))
            offset = -0.2 if lbl == 'Théo' else 0.2
            ax6.bar(x + offset, grp.get('wins', 0),  width=0.18, color=col_w, alpha=0.8)
            ax6.bar(x + offset, -grp.get('losses', 0), width=0.18, color=col_l, alpha=0.8)
    ax6.axhline(0, color=GREY, lw=0.7)
    ax6.set_xlabel('Mois', fontsize=8, color=GREY)
    ax6.set_ylabel('Nb trades', fontsize=8, color=GREY)
    ax6.grid(True, alpha=0.12, axis='y')
    ax6.tick_params(axis='x', labelsize=6, rotation=45)

    # ── 7. Z-scores dans le temps
    ax7 = fig.add_axes([0.01, 0.05, 0.97, 0.26])
    ax7.set_facecolor(BG2)
    _section_title(ax7, '📡  Z-SCORES AU FIL DU TEMPS — Signaux d\'entrée (▼) et de sortie (▲)')

    sig = rb['signals_df']
    ax7.scatter(sig.index, sig['z_score'], s=1.5, alpha=0.2, color=BLUE, zorder=1)

    # Zones colorées entrée/sortie
    ax7.axhspan(CONFIG['zscore_entry'] - 3, CONFIG['zscore_entry'],
                alpha=0.06, color=GREEN, label='Zone LONG')
    ax7.axhspan(CONFIG['zscore_exit'], CONFIG['zscore_exit'] + 3,
                alpha=0.06, color=RED, label='Zone EXIT')

    ax7.axhline(CONFIG['zscore_entry'], color=GREEN, lw=1.2, ls='--',
                label=f"Seuil entrée ({CONFIG['zscore_entry']})")
    ax7.axhline(CONFIG['zscore_exit'],  color=RED,   lw=1.2, ls='--',
                label=f"Seuil sortie ({CONFIG['zscore_exit']})")
    ax7.axhline(0, color=GREY, lw=0.6, alpha=0.5)

    # Marqueurs des trades
    if len(tb):
        entries = tb['entry_date']
        exits   = tb['exit_date']
        for _, row in tb.iterrows():
            if row['entry_date'] in sig.index:
                z_in = sig.loc[row['entry_date'], 'z_score'] if row['entry_date'] in sig.index else CONFIG['zscore_entry']
                ax7.scatter(row['entry_date'], z_in, marker='v', s=40,
                            color=GREEN, zorder=4, alpha=0.9)
            if row['exit_date'] in sig.index:
                z_out = sig.loc[row['exit_date'], 'z_score'] if row['exit_date'] in sig.index else CONFIG['zscore_exit']
                ax7.scatter(row['exit_date'], z_out, marker='^', s=40,
                            color=RED, zorder=4, alpha=0.9)

    ax7.set_ylabel('Z-score', color=GREY, fontsize=9)
    ax7.legend(fontsize=8, loc='upper right', ncol=3)
    ax7.grid(True, alpha=0.12)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax7.tick_params(labelsize=8)

    plt.savefig('spx_page3_trades.png', dpi=150, bbox_inches='tight', facecolor=BG)
    print('[OUTPUT] ✅  spx_page3_trades.png')
    plt.close()

# =============================================================================
# 11. PAGE 4 — OPTIMISATION
# =============================================================================

def plot_page4_optimization(optim_df, r_best):
    if optim_df is None or len(optim_df) == 0:
        print('[SKIP] Optimisation désactivée, page 4 non générée.')
        return

    apply_dark_style()
    fig = plt.figure(figsize=(20, 13), facecolor=BG)

    hax = fig.add_axes([0, 0.95, 1, 0.05])
    hax.axis('off')
    hax.text(0.5, 0.6, 'PAGE 4 — OPTIMISATION DES PARAMÈTRES (GRID SEARCH)',
             ha='center', fontsize=15, fontweight='bold', color=WHITE)
    hax.axhline(0.05, color=ACCENT, lw=1.5, alpha=0.6)

    best = optim_df.iloc[0]

    # ── Cartes best params ──────────────────────────────────────────────
    bax = fig.add_axes([0, 0.88, 1, 0.07])
    bax.axis('off')
    bax.set_facecolor(BG)
    bax.text(0.5, 0.8,
             f"★  MEILLEURE COMBINAISON  ·  Lookback = {int(best.lookback)}h  ·  "
             f"Entry Z = {best.entry_z}  ·  Exit Z = {best.exit_z}  ·  "
             f"Sharpe = {best.sharpe:.2f}  ·  CAGR = {best.cagr:+.2f}%  ·  "
             f"Max DD = {best.max_dd:.2f}%",
             ha='center', va='center', fontsize=10, fontweight='bold',
             color=YELLOW, transform=bax.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL, edgecolor=YELLOW, lw=1.5))

    # ── Heatmaps Sharpe pour chaque lookback ────────────────────────────
    lbs = sorted(optim_df['lookback'].unique())
    n_lb = len(lbs)
    for i, lb in enumerate(lbs):
        ax = fig.add_axes([0.01 + i * (0.98/n_lb), 0.57, (0.95/n_lb) - 0.01, 0.28])
        ax.set_facecolor(BG2)
        sub = optim_df[optim_df['lookback'] == lb]
        pivot = sub.pivot_table(values='sharpe', index='entry_z', columns='exit_z')
        if not pivot.empty:
            vmin = optim_df['sharpe'].min()
            vmax = optim_df['sharpe'].max()
            im = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                             ax=ax, linewidths=0.5, linecolor=BG,
                             vmin=vmin, vmax=vmax,
                             cbar=False,
                             annot_kws={'size': 8, 'weight': 'bold'})
            title_color = YELLOW if lb == best.lookback else WHITE
            ax.set_title(f'Lookback = {int(lb)}h — SHARPE',
                         fontsize=9, color=title_color, fontweight='bold', pad=6)
            ax.set_xlabel('Exit Z-score', fontsize=8, color=GREY)
            ax.set_ylabel('Entry Z-score', fontsize=8, color=GREY)
            ax.tick_params(labelsize=7)

            if lb == best.lookback:
                # Encadrer la meilleure cellule
                entry_idx = list(pivot.index).index(best.entry_z)
                exit_idx  = list(pivot.columns).index(best.exit_z)
                ax.add_patch(plt.Rectangle((exit_idx, entry_idx), 1, 1,
                             fill=False, edgecolor=YELLOW, lw=2.5))

    # ── Top 10 tableau ──────────────────────────────────────────────────
    ax_top = fig.add_axes([0.01, 0.27, 0.55, 0.27])
    ax_top.axis('off')
    ax_top.set_facecolor(BG2)
    ax_top.set_title('🏆  TOP 10 COMBINAISONS — triées par Sharpe',
                     fontsize=10, color=WHITE, pad=10, loc='left')

    top10 = optim_df.head(10)
    cols  = ['lookback', 'entry_z', 'exit_z', 'sharpe', 'cagr', 'max_dd', 'win_rate', 'n_trades']
    hdrs  = ['LOOKBACK', 'ENTRY Z', 'EXIT Z', 'SHARPE', 'CAGR%', 'MAX DD%', 'WIN%', 'N TRADES']
    col_x_t = [0.01, 0.14, 0.24, 0.34, 0.45, 0.56, 0.67, 0.78]

    for j, hdr in enumerate(hdrs):
        ax_top.text(col_x_t[j], 0.95, hdr, fontsize=7.5, fontweight='bold',
                    color=ACCENT, transform=ax_top.transAxes)
    ax_top.plot([0, 1], [0.91, 0.91], color=GREY, lw=0.4, transform=ax_top.transAxes, clip_on=False)

    for i, (_, row) in enumerate(top10.iterrows()):
        y = 0.86 - i * 0.085
        row_color = YELLOW if i == 0 else WHITE
        bg = PANEL if i % 2 == 0 else BG2
        rect = mpatches.FancyBboxPatch((0, y - 0.04), 1, 0.08,
                                        boxstyle="round,pad=0.005",
                                        facecolor=bg, edgecolor='none',
                                        transform=ax_top.transAxes, clip_on=False)
        ax_top.add_patch(rect)
        vals = [f"{int(row.lookback)}h", f"{row.entry_z}", f"{row.exit_z}",
                f"{row.sharpe:.2f}", f"{row.cagr:+.1f}%",
                f"{row.max_dd:.1f}%", f"{row.win_rate:.0f}%", f"{int(row.n_trades)}"]
        for j, val in enumerate(vals):
            ax_top.text(col_x_t[j], y, val, fontsize=8, color=row_color,
                        transform=ax_top.transAxes, va='center', fontfamily='monospace')

    # ── Scatter Sharpe vs CAGR ──────────────────────────────────────────
    ax_sc = fig.add_axes([0.59, 0.27, 0.40, 0.27])
    ax_sc.set_facecolor(BG2)
    _section_title(ax_sc, 'Sharpe vs CAGR — toutes combinaisons')

    sc = ax_sc.scatter(optim_df['cagr'], optim_df['sharpe'],
                       c=optim_df['max_dd'], cmap='RdYlGn_r',
                       s=50, alpha=0.7, edgecolors='none',
                       vmin=optim_df['max_dd'].min(), vmax=optim_df['max_dd'].max())
    plt.colorbar(sc, ax=ax_sc, label='Max DD (%)')
    ax_sc.scatter(best.cagr, best.sharpe, s=150, color=YELLOW,
                  marker='*', zorder=5, label=f'★ Meilleur (LB={int(best.lookback)}h)')
    ax_sc.axhline(0, color=GREY, lw=0.7, ls='--')
    ax_sc.axvline(0, color=GREY, lw=0.7, ls='--')
    ax_sc.set_xlabel('CAGR (%)', fontsize=9, color=GREY)
    ax_sc.set_ylabel('Sharpe', fontsize=9, color=GREY)
    ax_sc.legend(fontsize=8)
    ax_sc.grid(True, alpha=0.12)

    # ── Distribution Sharpe ──────────────────────────────────────────────
    ax_dist = fig.add_axes([0.01, 0.05, 0.47, 0.18])
    ax_dist.set_facecolor(BG2)
    _section_title(ax_dist, 'Distribution des Sharpes — toutes combinaisons')
    ax_dist.hist(optim_df['sharpe'], bins=20, color=BLUE, alpha=0.7, edgecolor='none')
    ax_dist.axvline(best.sharpe, color=YELLOW, lw=1.5, ls='--',
                    label=f'Meilleur ({best.sharpe:.2f})')
    ax_dist.axvline(optim_df['sharpe'].mean(), color=GREEN, lw=1, ls=':',
                    label=f'Moyenne ({optim_df["sharpe"].mean():.2f})')
    ax_dist.set_xlabel('Sharpe', fontsize=8, color=GREY)
    ax_dist.set_ylabel('Fréquence', fontsize=8, color=GREY)
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.12)

    # ── Distribution CAGR ───────────────────────────────────────────────
    ax_dcagr = fig.add_axes([0.52, 0.05, 0.47, 0.18])
    ax_dcagr.set_facecolor(BG2)
    _section_title(ax_dcagr, 'Distribution des CAGR — toutes combinaisons')
    bar_c = [GREEN if v >= 0 else RED for v in optim_df['cagr']]
    ax_dcagr.hist(optim_df['cagr'], bins=20, color=BLUE, alpha=0.7, edgecolor='none')
    ax_dcagr.axvline(0, color=GREY, lw=0.8, ls='--')
    ax_dcagr.axvline(best.cagr, color=YELLOW, lw=1.5, ls='--',
                     label=f'Meilleur ({best.cagr:+.1f}%)')
    ax_dcagr.set_xlabel('CAGR (%)', fontsize=8, color=GREY)
    ax_dcagr.set_ylabel('Fréquence', fontsize=8, color=GREY)
    ax_dcagr.legend(fontsize=8)
    ax_dcagr.grid(True, alpha=0.12)

    plt.savefig('spx_page4_optimization.png', dpi=150, bbox_inches='tight', facecolor=BG)
    print('[OUTPUT] ✅  spx_page4_optimization.png')
    plt.close()

# =============================================================================
# 12. PAGE 5 — TOP N COMBINAISONS COMPARÉES
# =============================================================================

def plot_page5_top_combinations(top_results):
    """Page dédiée : equity curves + métriques des N meilleures combinaisons."""
    if not top_results:
        return

    apply_dark_style()
    N   = len(top_results)
    TOP_COLORS = [YELLOW, GREEN, BLUE, ACCENT, RED]
    fig = plt.figure(figsize=(20, 14), facecolor=BG)

    # ── Header ──────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 0.95, 1, 0.05])
    hax.axis('off')
    hax.set_facecolor(BG)
    hax.text(0.5, 0.6,
             f'PAGE 5 — TOP {N} COMBINAISONS  ·  Capital de départ : ${top_results[0]["initial_capital"]:,.0f}',
             ha='center', fontsize=15, fontweight='bold', color=WHITE)
    hax.axhline(0.05, color=ACCENT, lw=1.5, alpha=0.6)

    # ── Equity curves superposées ────────────────────────────────────────
    ax_eq = fig.add_axes([0.01, 0.62, 0.63, 0.30])
    ax_eq.set_facecolor(BG2)
    _section_title(ax_eq, '📈  EQUITY CURVES COMPARÉES — Toutes avec frais CFD')

    for i, r in enumerate(top_results):
        col   = TOP_COLORS[i % len(TOP_COLORS)]
        eq    = r['equity_df']['equity']
        pct   = eq / r['initial_capital'] * 100 - 100
        lw    = 2.5 if i == 0 else 1.2
        alpha = 1.0 if i == 0 else 0.65
        ax_eq.plot(pct.index, pct.values, color=col, lw=lw, alpha=alpha,
                   label=f"#{i+1} LB={int(r.get('lookback_hours', '?'))}h  Sharpe={r['sharpe']:.2f}")

    ax_eq.axhline(0, color=GREY, lw=0.8, ls='--', alpha=0.5)
    ax_eq.set_ylabel('Return (%)', color=GREY, fontsize=9)
    ax_eq.legend(fontsize=8, loc='upper left')
    ax_eq.grid(True, alpha=0.15)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_eq.tick_params(labelsize=8)

    # ── Drawdowns comparés ───────────────────────────────────────────────
    ax_dd = fig.add_axes([0.66, 0.62, 0.33, 0.30])
    ax_dd.set_facecolor(BG2)
    _section_title(ax_dd, '📉  DRAWDOWNS COMPARÉS')

    for i, r in enumerate(top_results):
        col = TOP_COLORS[i % len(TOP_COLORS)]
        eq  = r['equity_df']['equity']
        dd  = (eq - eq.cummax()) / eq.cummax() * 100
        lw  = 2.0 if i == 0 else 1.0
        ax_dd.plot(dd.index, dd.values, color=col, lw=lw,
                   alpha=1.0 if i == 0 else 0.55,
                   label=f"#{i+1}")

    ax_dd.set_ylabel('Drawdown (%)', color=GREY, fontsize=9)
    ax_dd.legend(fontsize=8)
    ax_dd.grid(True, alpha=0.15)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_dd.tick_params(labelsize=8)

    # ── Tableau métriques détaillé ───────────────────────────────────────
    ax_tab = fig.add_axes([0.01, 0.03, 0.97, 0.56])
    ax_tab.axis('off')
    ax_tab.set_facecolor(BG)
    _section_title(ax_tab, f'📋  TABLEAU COMPARATIF — TOP {N} COMBINAISONS (avec frais CFD, ${top_results[0]["initial_capital"]:,.0f} de départ)')

    metrics_rows = [
        ('Return Total (3 ans)',  lambda r: f"{r['total_return_pct']:+.2f}%"),
        ('CAGR (annualisé)',       lambda r: f"{r['cagr_pct']:+.2f}%"),
        ('Sharpe Ratio',           lambda r: f"{r['sharpe']:.2f}"),
        ('Calmar Ratio',           lambda r: f"{r['calmar']:.2f}"),
        ('Max Drawdown',           lambda r: f"{r['max_drawdown_pct']:.2f}%"),
        ('Nombre de trades',       lambda r: f"{r['n_trades']}"),
        ('Win Rate',               lambda r: f"{r['win_rate_pct']:.1f}%"),
        ('Profit Factor',          lambda r: f"{r['profit_factor']:.2f}"),
        ('Avg Win',                lambda r: f"{r['avg_win_pct']:.3f}%"),
        ('Avg Loss',               lambda r: f"{r['avg_loss_pct']:.3f}%"),
        ('Expectancy',             lambda r: f"{r['expectancy']:.3f}%"),
        ('Durée moy. trade (j)',   lambda r: f"{r['avg_days_held']:.1f}j"),
        (f'Capital final ($)',      lambda r: f"${r['final_capital']:,.0f}"),
        (f'Gain net ($)',           lambda r: f"${r['final_capital'] - r['initial_capital']:+,.0f}"),
    ]

    # En-têtes colonnes
    col_x = [0.01] + [0.22 + i * (0.77 / N) for i in range(N)]
    ax_tab.text(col_x[0], 0.93, 'MÉTRIQUE', fontsize=8.5, fontweight='bold',
                color=WHITE, transform=ax_tab.transAxes)
    for i, r in enumerate(top_results):
        col = TOP_COLORS[i % len(TOP_COLORS)]
        medal = ['🥇','🥈','🥉','4️⃣','5️⃣'][i] if i < 5 else f'#{i+1}'
        ax_tab.text(col_x[i+1], 0.93, f"{medal} #{i+1}", fontsize=8.5,
                    fontweight='bold', color=col, transform=ax_tab.transAxes)

    ax_tab.plot([0, 1], [0.90, 0.90], color=GREY, lw=0.5, transform=ax_tab.transAxes, clip_on=False)

    for row_i, (label, fn) in enumerate(metrics_rows):
        y = 0.86 - row_i * 0.060
        bg = PANEL if row_i % 2 == 0 else BG2
        rect = mpatches.FancyBboxPatch((0, y - 0.025), 1, 0.052,
                                        boxstyle="round,pad=0.005",
                                        facecolor=bg, edgecolor='none',
                                        transform=ax_tab.transAxes, clip_on=False)
        ax_tab.add_patch(rect)
        ax_tab.text(col_x[0], y, label, fontsize=8, color=WHITE,
                    transform=ax_tab.transAxes, va='center')
        for i, r in enumerate(top_results):
            col = TOP_COLORS[i % len(TOP_COLORS)]
            val = fn(r)
            # Couleur conditionnelle pour certaines métriques
            if 'Return' in label or 'CAGR' in label or 'Gain' in label or 'Capital' in label:
                disp_col = GREEN if r['total_return_pct'] >= 0 else RED
            elif 'Drawdown' in label:
                disp_col = RED
            elif 'Sharpe' in label or 'Win' in label or 'Factor' in label:
                disp_col = GREEN if r['sharpe'] >= 1 else YELLOW
            else:
                disp_col = col
            ax_tab.text(col_x[i+1], y, val, fontsize=8, color=disp_col,
                        transform=ax_tab.transAxes, va='center', fontfamily='monospace')

    # Paramètres de chaque combi en sous-titre de colonne
    for i, r in enumerate(top_results):
        col = TOP_COLORS[i % len(TOP_COLORS)]
        # On récupère les params depuis le label
        lbl_short = r['label'].split('|')
        params_txt = '  '.join(p.strip() for p in lbl_short[1:]) if len(lbl_short) > 1 else r['label']
        ax_tab.text(col_x[i+1], 0.97, params_txt, fontsize=6.5, color=col,
                    transform=ax_tab.transAxes, va='top', style='italic')

    filename = f'spx_page5_top{len(top_results)}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'[OUTPUT] ✅  {filename}')
    plt.close()


# =============================================================================
# 13. EXPORT CSV
# =============================================================================

def export_results(rb, rc, optim_df=None):
    rb['trades_df'].to_csv('trades_brut.csv', index=False)
    rc['trades_df'].to_csv('trades_cfd.csv',  index=False)
    rb['equity_df'].to_csv('equity_brut.csv')
    rc['equity_df'].to_csv('equity_cfd.csv')
    summary = pd.DataFrame([
        {k: v for k, v in rb.items() if not isinstance(v, pd.DataFrame)},
        {k: v for k, v in rc.items() if not isinstance(v, pd.DataFrame)},
    ])
    summary.to_csv('metrics_summary.csv', index=False)
    if optim_df is not None and len(optim_df) > 0:
        optim_df.to_csv('optimization_results.csv', index=False)
        print('[OUTPUT] optimization_results.csv')
    print('[OUTPUT] trades_brut.csv | trades_cfd.csv')
    print('[OUTPUT] equity_brut.csv | equity_cfd.csv')
    print('[OUTPUT] metrics_summary.csv')

# =============================================================================
# 13. POINT D'ENTRÉE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SPX/USD — MEAN REVERSION BACKTEST")
    print(f"  Capital de départ : ${CONFIG['initial_capital']:,.0f}")
    print("  Mode : TOP COMBINAISONS UNIQUEMENT")
    print("=" * 60)

    df = load_data(CONFIG['data_path'])

    # ── Optimisation (obligatoire dans ce mode) ──
    CONFIG['optimize'] = True
    optim_df = run_optimization(df, CONFIG)

    TOP_N = CONFIG.get('top_n_combinations', 5)
    top_combos = optim_df.head(TOP_N)

    print(f"\n🏆 TOP {TOP_N} COMBINAISONS sélectionnées pour analyse approfondie :")
    print(top_combos[['lookback','entry_z','exit_z','sharpe','cagr','max_dd','win_rate','n_trades']].to_string(index=False))

    # ── Backtests détaillés sur le TOP N (avec frais CFD) ──
    top_results = []
    for rank, (_, row) in enumerate(top_combos.iterrows(), start=1):
        lbl = f"TOP{rank} | LB={int(row.lookback)}h | Z={row.entry_z}/{row.exit_z}"
        r = run_backtest(df,
            lookback_hours=int(row.lookback),
            zscore_entry=row.entry_z,
            zscore_exit=row.exit_z,
            adf_pvalue=CONFIG['adf_pvalue'],
            initial_capital=CONFIG['initial_capital'],
            spread_points=CONFIG['spread_points'],
            swap_rate_annual=CONFIG['swap_rate_annual'],
            commission=CONFIG['commission_per_trade'],
            market_open_hour=CONFIG['market_open_hour'],
            market_open_minute=CONFIG['market_open_minute'],
            label=lbl
        )
        r['rank'] = rank
        top_results.append(r)

    rb     = top_results[0]   # meilleure combinaison = référence "brut" pour pages 2/3
    rc     = top_results[1] if len(top_results) > 1 else top_results[0]
    r_best = top_results[0]

    # ── Backtest théorique (sans frais) de la #1 pour page 1 ──
    rb_no_cost = run_backtest(df,
        lookback_hours=int(top_combos.iloc[0].lookback),
        zscore_entry=top_combos.iloc[0].entry_z,
        zscore_exit=top_combos.iloc[0].exit_z,
        adf_pvalue=CONFIG['adf_pvalue'],
        initial_capital=CONFIG['initial_capital'],
        spread_points=0.0, swap_rate_annual=0.0, commission=0.0,
        market_open_hour=CONFIG['market_open_hour'],
        market_open_minute=CONFIG['market_open_minute'],
        label="THÉORIQUE — Meilleure combi, sans frais"
    )

    # ── 4 pages de graphiques ──
    print("\n[CHARTS] Génération des graphiques...")
    plot_page1_overview(rb_no_cost, rb, r_best)
    plot_page2_performance(rb_no_cost, rb)
    plot_page3_trades(rb_no_cost, rb)
    plot_page4_optimization(optim_df, r_best)
    plot_page5_top_combinations(top_results)

    # ── Exports CSV ──
    export_results(rb_no_cost, rb, optim_df)

    print("\n" + "=" * 60)
    print("  ✅  TERMINÉ")
    print(f"  Capital de départ : ${CONFIG['initial_capital']:,.0f}")
    print(f"  Capital final #1  : ${rb['final_capital']:>12,.0f}  (CFD avec frais)")
    print(f"  Return total #1   : {rb['total_return_pct']:+.2f}%")
    print(f"  Sharpe #1         : {rb['sharpe']:.2f}")
    print("  Fichiers générés :")
    print("    📊 spx_page1_overview.png      ← Vue d'ensemble + tableau comparatif")
    print("    📈 spx_page2_performance.png   ← Equity, drawdown, returns mensuels/annuels")
    print("    🔍 spx_page3_trades.png        ← Analyse des trades + z-scores")
    print("    🏆 spx_page4_optimization.png  ← Grid search + heatmaps")
    print(f"    🥇 spx_page5_top{TOP_N}.png         ← Comparatif TOP {TOP_N} combinaisons")
    print("    📋 CSV : trades, equity, metrics, optimization")
    print("=" * 60)