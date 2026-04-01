"""
plotting.py — Génération des 4 pages de visualisations.
cfg est passé explicitement dans chaque fonction (plus de CONFIG global).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

from config import Config

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0B0E1A"
BG2    = "#131729"
PANEL  = "#1A1F35"
GREEN  = "#00E5A0"
RED    = "#FF4C6E"
YELLOW = "#FFD166"
BLUE   = "#4EA8DE"
GREY   = "#4A5568"
WHITE  = "#E8EDF5"
ACCENT = "#7C5CFC"


def apply_dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": PANEL, "axes.labelcolor": WHITE,
        "xtick.color": GREY, "ytick.color": GREY,
        "text.color": WHITE, "grid.color": PANEL,
        "grid.linewidth": 0.6, "legend.facecolor": PANEL,
        "legend.edgecolor": GREY, "font.family": "monospace",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _section_title(ax, text: str, color: str = WHITE) -> None:
    ax.set_title(text, fontsize=11, fontweight="bold", color=color, pad=10, loc="left")


def _metric_card(fig, rect, title: str, value: str,
                 subtitle: str = "", color: str = GREEN, fontsize_val: int = 22):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.72, title, ha="center", va="center",
            fontsize=7.5, color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.42, value, ha="center", va="center",
            fontsize=fontsize_val, fontweight="bold", color=color, transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.14, subtitle, ha="center", va="center",
                fontsize=7, color=GREY, transform=ax.transAxes)
    return ax


def plot_page1_overview(rb: dict, rc: dict, cfg: Config, r_best: dict = None) -> None:
    apply_dark_style()
    fig = plt.figure(figsize=(20, 13), facecolor=BG)

    hax = fig.add_axes([0, 0.91, 1, 0.09])
    hax.set_facecolor(BG); hax.axis("off")
    hax.text(0.5, 0.72, "SPX/USD — MEAN REVERSION STRATEGY",
             ha="center", va="center", fontsize=20, fontweight="bold", color=WHITE)
    hax.text(0.5, 0.22,
             "ADF Stationarity Filter · Z-Score Signal · Long Only · 3 ans de données",
             ha="center", va="center", fontsize=10, color=GREY)
    hax.axhline(0.02, color=ACCENT, linewidth=1.5, alpha=0.6)

    lax = fig.add_axes([0, 0.87, 1, 0.04])
    lax.axis("off"); lax.set_facecolor(BG)
    for x, col, txt in [
        (0.18, GREEN, "● THÉORIQUE — Sans frais, potentiel brut maximum"),
        (0.55, RED,   "● SIMULATION CFD — Spread + financement overnight"),
    ]:
        lax.text(x, 0.5, txt, ha="left", va="center", fontsize=9, color=col, fontweight="bold")
    if r_best:
        lax.text(0.82, 0.5, "★ PARAMÈTRES OPTIMAUX", ha="left", va="center",
                 fontsize=9, color=YELLOW, fontweight="bold")

    for i, (title, val, sub) in enumerate([
        ("RETURN TOTAL",  f"{rb['total_return_pct']:+.1f}%", "3 ans"),
        ("CAGR",          f"{rb['cagr_pct']:+.1f}%",         "par an"),
        ("SHARPE",        f"{rb['sharpe']:.2f}",              "risk-adjusted"),
        ("MAX DRAWDOWN",  f"{rb['max_drawdown_pct']:.1f}%",   "pire période"),
        ("WIN RATE",      f"{rb['win_rate_pct']:.0f}%",       f"{rb['n_trades']} trades"),
        ("PROFIT FACTOR", f"{rb['profit_factor']:.2f}",       "wins/losses"),
    ]):
        col = RED if "DRAWDOWN" in title else GREEN
        _metric_card(fig, [0.01 + i * 0.165, 0.72, 0.155, 0.14], title, val, sub, color=col)

    for i, (title, val, sub) in enumerate([
        ("RETURN TOTAL",  f"{rc['total_return_pct']:+.1f}%", "3 ans"),
        ("CAGR",          f"{rc['cagr_pct']:+.1f}%",         "par an"),
        ("SHARPE",        f"{rc['sharpe']:.2f}",              "risk-adjusted"),
        ("MAX DRAWDOWN",  f"{rc['max_drawdown_pct']:.1f}%",   "pire période"),
        ("WIN RATE",      f"{rc['win_rate_pct']:.0f}%",       f"{rc['n_trades']} trades"),
        ("PROFIT FACTOR", f"{rc['profit_factor']:.2f}",       "wins/losses"),
    ]):
        _metric_card(fig, [0.01 + i * 0.165, 0.56, 0.155, 0.14], title, val, sub, color=RED)

    ax_eq = fig.add_axes([0.01, 0.28, 0.63, 0.26])
    ax_eq.set_facecolor(BG2)
    _section_title(ax_eq, "📈 PERFORMANCE CUMULÉE — Les 3 scénarios comparés")
    eq_b = rb["equity_df"]["equity"] / rb["initial_capital"] * 100 - 100
    eq_c = rc["equity_df"]["equity"] / rc["initial_capital"] * 100 - 100
    ax_eq.plot(eq_b.index, eq_b.values, color=GREEN, lw=2, label="Théorique", zorder=3)
    ax_eq.plot(eq_c.index, eq_c.values, color=RED,   lw=2, label="CFD",       zorder=3)
    ax_eq.fill_between(eq_b.index, eq_b.values, 0, where=(eq_b.values >= 0), alpha=0.08, color=GREEN)
    ax_eq.fill_between(eq_b.index, eq_b.values, 0, where=(eq_b.values < 0),  alpha=0.08, color=RED)
    ax_eq.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.5)
    if r_best:
        eq_best = r_best["equity_df"]["equity"] / r_best["initial_capital"] * 100 - 100
        ax_eq.plot(eq_best.index, eq_best.values, color=YELLOW, lw=1.5,
                   ls="--", label="Paramètres optimaux (CFD)", zorder=4)
    ax_eq.set_ylabel("Return (%)", color=GREY, fontsize=9)
    ax_eq.legend(fontsize=8, loc="upper left")
    ax_eq.grid(True, alpha=0.15)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_eq.tick_params(labelsize=8)

    ax_dd = fig.add_axes([0.66, 0.28, 0.33, 0.26])
    ax_dd.set_facecolor(BG2)
    _section_title(ax_dd, "📉 DRAWDOWN")
    for res, color, lbl in [(rb, GREEN, "Théorique"), (rc, RED, "CFD")]:
        eq = res["equity_df"]["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.45, color=color, label=lbl)
        ax_dd.plot(dd.index, dd.values, color=color, lw=0.8, alpha=0.7)
    ax_dd.set_ylabel("Drawdown (%)", color=GREY, fontsize=9)
    ax_dd.legend(fontsize=8)
    ax_dd.grid(True, alpha=0.15)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_dd.tick_params(labelsize=8)

    ax_tab = fig.add_axes([0.01, 0.03, 0.97, 0.23])
    ax_tab.axis("off"); ax_tab.set_facecolor(BG)
    ax_tab.text(0.0, 0.97, "📋 TABLEAU COMPARATIF COMPLET",
                fontsize=10, fontweight="bold", color=WHITE,
                transform=ax_tab.transAxes, va="top")

    rows = [
        ("Return Total (3 ans)",
         f"{rb['total_return_pct']:+.2f}%", f"{rc['total_return_pct']:+.2f}%",
         f"{r_best['total_return_pct']:+.2f}%" if r_best else "—"),
        ("CAGR (annualisé)",
         f"{rb['cagr_pct']:+.2f}%", f"{rc['cagr_pct']:+.2f}%",
         f"{r_best['cagr_pct']:+.2f}%" if r_best else "—"),
        ("Ratio de Sharpe",
         f"{rb['sharpe']:.2f}", f"{rc['sharpe']:.2f}",
         f"{r_best['sharpe']:.2f}" if r_best else "—"),
        ("Ratio de Calmar",
         f"{rb['calmar']:.2f}", f"{rc['calmar']:.2f}",
         f"{r_best['calmar']:.2f}" if r_best else "—"),
        ("Max Drawdown",
         f"{rb['max_drawdown_pct']:.2f}%", f"{rc['max_drawdown_pct']:.2f}%",
         f"{r_best['max_drawdown_pct']:.2f}%" if r_best else "—"),
        ("Nombre de trades",
         str(rb["n_trades"]), str(rc["n_trades"]),
         str(r_best["n_trades"]) if r_best else "—"),
        ("Win Rate",
         f"{rb['win_rate_pct']:.1f}%", f"{rc['win_rate_pct']:.1f}%",
         f"{r_best['win_rate_pct']:.1f}%" if r_best else "—"),
        ("Profit Factor",
         f"{rb['profit_factor']:.2f}", f"{rc['profit_factor']:.2f}",
         f"{r_best['profit_factor']:.2f}" if r_best else "—"),
        ("Capital final ($1M départ)",
         f"${rb['final_capital']:>10,.0f}", f"${rc['final_capital']:>10,.0f}",
         f"${r_best['final_capital']:>10,.0f}" if r_best else "—"),
    ]

    col_x = [0.01, 0.32, 0.55, 0.78]
    headers = ["MÉTRIQUE", "🟢 THÉORIQUE", "🔴 SIMULATION CFD", "⭐ OPTIMAUX"]
    h_colors = [WHITE, GREEN, RED, YELLOW]
    for j, (hdr, hcol) in enumerate(zip(headers, h_colors)):
        ax_tab.text(col_x[j], 0.87, hdr, fontsize=8.5, fontweight="bold",
                    color=hcol, transform=ax_tab.transAxes, va="top")
    ax_tab.plot([0, 1], [0.83, 0.83], color=GREY, lw=0.5,
                transform=ax_tab.transAxes, clip_on=False)
    for i, row in enumerate(rows):
        y = 0.78 - i * 0.075
        bg_c = PANEL if i % 2 == 0 else BG2
        rect = mpatches.FancyBboxPatch((0, y - 0.025), 1, 0.063,
                                        boxstyle="round,pad=0.005",
                                        facecolor=bg_c, edgecolor="none",
                                        transform=ax_tab.transAxes, clip_on=False)
        ax_tab.add_patch(rect)
        for j, val in enumerate(row):
            col = WHITE if j == 0 else (GREEN if j == 1 else (RED if j == 2 else YELLOW))
            ax_tab.text(col_x[j], y, val, fontsize=8, color=col,
                        transform=ax_tab.transAxes, va="center", fontfamily="monospace")

    plt.savefig("spx_page1_overview.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("[OUTPUT] ✅ spx_page1_overview.png")
    plt.close()


def plot_page2_performance(rb: dict, rc: dict) -> None:
    apply_dark_style()
    fig = plt.figure(figsize=(20, 14), facecolor=BG)

    hax = fig.add_axes([0, 0.95, 1, 0.05]); hax.axis("off")
    hax.text(0.5, 0.6, "PAGE 2 — ANALYSE DE PERFORMANCE DÉTAILLÉE",
             ha="center", fontsize=15, fontweight="bold", color=WHITE)
    hax.axhline(0.05, color=ACCENT, lw=1.5, alpha=0.6)

    scenarios = [
        (rb, GREEN, "🟢 THÉORIQUE — Sans aucun frais",
         "Ce que la stratégie ferait dans un monde idéal sans coûts"),
        (rc, RED, "🔴 SIMULATION CFD — Spread 0.5pt + swap 5%/an",
         "Ce que tu obtiendrais réellement en tradant ce signal sur un CFD SPX"),
    ]

    for idx_s, (res, color, title, subtitle) in enumerate(scenarios):
        top = 0.68 - idx_s * 0.38
        sax = fig.add_axes([0, top + 0.09, 1, 0.04]); sax.axis("off")
        sax.text(0.01, 0.5, title,     fontsize=11, fontweight="bold", color=color)
        sax.text(0.01, 0.05, subtitle, fontsize=8.5, color=GREY)
        eq = res["equity_df"]["equity"]

        ax1 = fig.add_axes([0.01, top, 0.44, 0.085]); ax1.set_facecolor(BG2)
        pct = eq / res["initial_capital"] * 100 - 100
        ax1.plot(pct.index, pct.values, color=color, lw=1.5)
        ax1.fill_between(pct.index, pct.values, 0, where=(pct.values >= 0), alpha=0.1, color=color)
        ax1.fill_between(pct.index, pct.values, 0, where=(pct.values < 0),  alpha=0.1, color=RED)
        ax1.axhline(0, color=GREY, lw=0.7, ls="--", alpha=0.5)
        ax1.set_title("Equity curve (% return)", fontsize=9, color=GREY, loc="left")
        ax1.grid(True, alpha=0.12)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.tick_params(labelsize=7)

        ax2 = fig.add_axes([0.47, top, 0.27, 0.085]); ax2.set_facecolor(BG2)
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.6, color=RED)
        ax2.plot(dd.index, dd.values, color=RED, lw=0.8)
        ax2.set_title("Drawdown (%)", fontsize=9, color=GREY, loc="left")
        ax2.grid(True, alpha=0.12)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.tick_params(labelsize=7)

        ax3 = fig.add_axes([0.76, top, 0.23, 0.085]); ax3.set_facecolor(BG2)
        monthly = eq.resample("ME").last().pct_change().dropna() * 100
        bar_colors = [GREEN if v >= 0 else RED for v in monthly.values]
        ax3.bar(range(len(monthly)), monthly.values, color=bar_colors, alpha=0.8, width=0.7)
        ax3.axhline(0, color=GREY, lw=0.7)
        ax3.set_title("Returns mensuels (%)", fontsize=9, color=GREY, loc="left")
        ax3.set_xticks(range(0, len(monthly), 3))
        ax3.set_xticklabels(
            [monthly.index[i].strftime("%m/%y") for i in range(0, len(monthly), 3)],
            fontsize=6, rotation=45)
        ax3.grid(True, alpha=0.12, axis="y")
        ax3.tick_params(labelsize=7)

    ax_yr = fig.add_axes([0.01, 0.04, 0.55, 0.15]); ax_yr.set_facecolor(BG2)
    ax_yr.set_title("📅 RETURNS ANNUELS COMPARÉS", fontsize=9, color=WHITE, loc="left", pad=8)
    for res, color, lbl in [(rb, GREEN, "Théorique"), (rc, RED, "CFD")]:
        eq = res["equity_df"]["equity"]
        yr = eq.resample("YE").last().pct_change().dropna() * 100
        yrs = [str(d.year) for d in yr.index]
        x = np.arange(len(yrs))
        offset = -0.2 if color == GREEN else 0.2
        bars = ax_yr.bar(x + offset, yr.values, width=0.38, color=color, alpha=0.8, label=lbl)
        for bar, val in zip(bars, yr.values):
            ax_yr.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + (0.3 if val >= 0 else -1.5),
                       f"{val:+.1f}%", ha="center", fontsize=7.5, color=color, fontweight="bold")
    ax_yr.set_xticks(np.arange(len(yrs))); ax_yr.set_xticklabels(yrs, fontsize=9)
    ax_yr.axhline(0, color=GREY, lw=0.7)
    ax_yr.legend(fontsize=8); ax_yr.grid(True, alpha=0.12, axis="y")

    ax_rs = fig.add_axes([0.59, 0.04, 0.40, 0.15]); ax_rs.set_facecolor(BG2)
    ax_rs.set_title("📊 ROLLING SHARPE (90 jours)", fontsize=9, color=WHITE, loc="left", pad=8)
    for res, color, lbl in [(rb, GREEN, "Théorique"), (rc, RED, "CFD")]:
        eq = res["equity_df"]["equity"]
        daily = eq.resample("1D").last().pct_change().dropna()
        rs = daily.rolling(90).mean() / daily.rolling(90).std() * np.sqrt(252)
        ax_rs.plot(rs.index, rs.values, color=color, lw=1.2, label=lbl)
    ax_rs.axhline(0, color=GREY, lw=0.7, ls="--")
    ax_rs.axhline(1, color=YELLOW, lw=0.7, ls=":", alpha=0.6, label="Sharpe=1")
    ax_rs.legend(fontsize=8); ax_rs.grid(True, alpha=0.12)
    ax_rs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_rs.tick_params(labelsize=7)

    plt.savefig("spx_page2_performance.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("[OUTPUT] ✅ spx_page2_performance.png")
    plt.close()


def plot_page3_trades(rb: dict, rc: dict, cfg: Config) -> None:
    apply_dark_style()
    fig = plt.figure(figsize=(20, 13), facecolor=BG)

    hax = fig.add_axes([0, 0.95, 1, 0.05]); hax.axis("off")
    hax.text(0.5, 0.6, "PAGE 3 — ANALYSE MICROSCOPIQUE DES TRADES",
             ha="center", fontsize=15, fontweight="bold", color=WHITE)
    hax.axhline(0.05, color=ACCENT, lw=1.5, alpha=0.6)

    tb = rb["trades_df"]; tc = rc["trades_df"]
    scenario_pairs = [(tb, GREEN, "Théorique"), (tc, RED, "CFD")]

    ax1 = fig.add_axes([0.01, 0.63, 0.30, 0.25]); ax1.set_facecolor(BG2)
    _section_title(ax1, "Distribution PnL par trade (%)")
    for t, col, lbl in scenario_pairs:
        if len(t):
            ax1.hist(t["pnl_pct"].dropna(), bins=25, alpha=0.6, color=col, label=lbl, edgecolor="none")
    ax1.axvline(0, color=WHITE, lw=1.2, ls="--", alpha=0.7)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.12)

    ax2 = fig.add_axes([0.34, 0.63, 0.30, 0.25]); ax2.set_facecolor(BG2)
    _section_title(ax2, "Distribution durée des trades (jours)")
    for t, col, lbl in scenario_pairs:
        if len(t) and "days_held" in t.columns:
            ax2.hist(t["days_held"].dropna(), bins=20, alpha=0.6, color=col, label=lbl, edgecolor="none")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.12)

    ax3 = fig.add_axes([0.67, 0.63, 0.32, 0.25]); ax3.set_facecolor(BG2)
    _section_title(ax3, "PnL cumulatif trade par trade")
    for t, col, lbl in scenario_pairs:
        if len(t):
            cumul = t["pnl"].dropna().cumsum()
            ax3.plot(range(len(cumul)), cumul.values, color=col, lw=1.5, label=lbl)
            ax3.fill_between(range(len(cumul)), cumul.values, 0,
                             where=(cumul.values >= 0), alpha=0.08, color=col)
    ax3.axhline(0, color=GREY, lw=0.7, ls="--")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.12)

    ax4 = fig.add_axes([0.01, 0.34, 0.30, 0.25]); ax4.set_facecolor(BG2)
    _section_title(ax4, "Z-score d'entrée vs PnL (%)")
    for t, col, lbl in scenario_pairs:
        if len(t):
            ax4.scatter(t["entry_zscore"], t["pnl_pct"],
                        color=col, alpha=0.5, s=18, label=lbl, edgecolors="none")
    ax4.axhline(0, color=GREY, lw=0.7, ls="--")
    ax4.axvline(cfg.zscore_entry, color=YELLOW, lw=0.8, ls=":", alpha=0.8)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.12)

    ax5 = fig.add_axes([0.34, 0.34, 0.30, 0.25]); ax5.set_facecolor(BG2)
    _section_title(ax5, "Durée (jours) vs PnL (%)")
    for t, col, lbl in scenario_pairs:
        if len(t) and "days_held" in t.columns:
            ax5.scatter(t["days_held"], t["pnl_pct"],
                        color=col, alpha=0.5, s=18, label=lbl, edgecolors="none")
    ax5.axhline(0, color=GREY, lw=0.7, ls="--")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.12)

    ax6 = fig.add_axes([0.67, 0.34, 0.32, 0.25]); ax6.set_facecolor(BG2)
    _section_title(ax6, "Trades gagnants vs perdants par mois")
    for t, col_w, col_l, lbl in [
        (tb, GREEN, "#1a6640", "Théo"),
        (tc, "#ff8fa3", RED,   "CFD"),
    ]:
        if len(t):
            t2 = t.copy()
            t2["month"] = pd.to_datetime(t2["entry_date"]).dt.to_period("M")
            grp = t2.groupby("month")["pnl"].apply(
                lambda x: pd.Series({"wins": (x > 0).sum(), "losses": (x <= 0).sum()})
            ).unstack()
            x_pos = np.arange(len(grp.index))
            offset = -0.2 if lbl == "Théo" else 0.2
            ax6.bar(x_pos + offset, grp.get("wins",   pd.Series([0])), width=0.18, color=col_w, alpha=0.8)
            ax6.bar(x_pos + offset, -grp.get("losses", pd.Series([0])), width=0.18, color=col_l, alpha=0.8)
    ax6.axhline(0, color=GREY, lw=0.7)
    ax6.grid(True, alpha=0.12, axis="y")
    ax6.tick_params(axis="x", labelsize=6, rotation=45)

    ax7 = fig.add_axes([0.01, 0.05, 0.97, 0.26]); ax7.set_facecolor(BG2)
    _section_title(ax7, "📡 Z-SCORES AU FIL DU TEMPS — Signaux d'entrée (▼) et de sortie (▲)")
    sig = rb["signals_df"]
    ax7.scatter(sig.index, sig["z_score"], s=1.5, alpha=0.2, color=BLUE, zorder=1)
    ax7.axhspan(cfg.zscore_entry - 3, cfg.zscore_entry, alpha=0.06, color=GREEN)
    ax7.axhspan(cfg.zscore_exit, cfg.zscore_exit + 3,   alpha=0.06, color=RED)
    ax7.axhline(cfg.zscore_entry, color=GREEN, lw=1.2, ls="--",
                label=f"Seuil entrée ({cfg.zscore_entry})")
    ax7.axhline(cfg.zscore_exit,  color=RED,   lw=1.2, ls="--",
                label=f"Seuil sortie ({cfg.zscore_exit})")
    ax7.axhline(0, color=GREY, lw=0.6, alpha=0.5)
    for _, row in tb.iterrows():
        if row["entry_date"] in sig.index:
            ax7.scatter(row["entry_date"], sig.loc[row["entry_date"], "z_score"],
                        marker="v", s=40, color=GREEN, zorder=4, alpha=0.9)
        if row["exit_date"] in sig.index:
            ax7.scatter(row["exit_date"], sig.loc[row["exit_date"], "z_score"],
                        marker="^", s=40, color=RED, zorder=4, alpha=0.9)
    ax7.legend(fontsize=8, loc="upper right", ncol=3)
    ax7.grid(True, alpha=0.12)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.savefig("spx_page3_trades.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("[OUTPUT] ✅ spx_page3_trades.png")
    plt.close()


def plot_page4_optimization(optim_df: pd.DataFrame, r_best: dict) -> None:
    if optim_df is None or len(optim_df) == 0:
        print("[SKIP] Optimisation désactivée, page 4 non générée.")
        return

    apply_dark_style()
    fig = plt.figure(figsize=(20, 13), facecolor=BG)

    hax = fig.add_axes([0, 0.95, 1, 0.05]); hax.axis("off")
    hax.text(0.5, 0.6, "PAGE 4 — OPTIMISATION DES PARAMÈTRES (GRID SEARCH)",
             ha="center", fontsize=15, fontweight="bold", color=WHITE)
    hax.axhline(0.05, color=ACCENT, lw=1.5, alpha=0.6)

    best = optim_df.iloc[0]
    bax = fig.add_axes([0, 0.88, 1, 0.07]); bax.axis("off")
    bax.text(
        0.5, 0.8,
        f"★ MEILLEURE COMBINAISON · Lookback={int(best.lookback)}h · "
        f"Entry Z={best.entry_z} · Exit Z={best.exit_z} · "
        f"Sharpe={best.sharpe:.2f} · CAGR={best.cagr:+.2f}% · MaxDD={best.max_dd:.2f}%",
        ha="center", va="center", fontsize=10, fontweight="bold", color=YELLOW,
        transform=bax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor=YELLOW, lw=1.5),
    )

    top_n = min(10, len(optim_df))

    ax1 = fig.add_axes([0.01, 0.52, 0.45, 0.33]); ax1.set_facecolor(BG2)
    _section_title(ax1, "Heatmap Sharpe : Lookback × Entry Z-score")
    pivot = optim_df.pivot_table(values="sharpe", index="lookback", columns="entry_z", aggfunc="max")
    sns.heatmap(pivot, ax=ax1, cmap="RdYlGn", annot=True, fmt=".2f",
                linewidths=0.5, linecolor=PANEL, cbar_kws={"shrink": 0.7})
    ax1.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.53, 0.52, 0.45, 0.33]); ax2.set_facecolor(BG2)
    _section_title(ax2, "Heatmap Sharpe : Entry Z × Exit Z")
    pivot2 = optim_df.pivot_table(values="sharpe", index="entry_z", columns="exit_z", aggfunc="max")
    sns.heatmap(pivot2, ax=ax2, cmap="RdYlGn", annot=True, fmt=".2f",
                linewidths=0.5, linecolor=PANEL, cbar_kws={"shrink": 0.7})
    ax2.tick_params(labelsize=8)

    ax3 = fig.add_axes([0.01, 0.14, 0.60, 0.33]); ax3.set_facecolor(BG2)
    _section_title(ax3, f"Top {top_n} combinaisons par Sharpe ratio")
    top = optim_df.head(top_n).copy()
    top["label_combo"] = top.apply(
        lambda r: f"LB={int(r.lookback)} Ze={r.entry_z} Zx={r.exit_z}", axis=1)
    bar_cols = [YELLOW if i == 0 else BLUE for i in range(top_n)]
    bars = ax3.barh(range(top_n), top["sharpe"].values, color=bar_cols, alpha=0.85)
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels(top["label_combo"].values, fontsize=8)
    ax3.set_xlabel("Sharpe Ratio", color=GREY, fontsize=8)
    ax3.axvline(0, color=GREY, lw=0.7); ax3.grid(True, alpha=0.12, axis="x")
    for bar, val in zip(bars, top["sharpe"].values):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=8, color=WHITE)

    ax4 = fig.add_axes([0.67, 0.14, 0.31, 0.33]); ax4.set_facecolor(BG2)
    _section_title(ax4, "CAGR vs Max Drawdown")
    sc = ax4.scatter(optim_df["max_dd"], optim_df["cagr"],
                     c=optim_df["sharpe"], cmap="RdYlGn", s=40, alpha=0.8, edgecolors="none")
    ax4.scatter(best.max_dd, best.cagr, color=YELLOW, s=120, zorder=5, marker="*")
    ax4.set_xlabel("Max Drawdown (%)", color=GREY, fontsize=8)
    ax4.set_ylabel("CAGR (%)", color=GREY, fontsize=8)
    ax4.grid(True, alpha=0.12)
    fig.colorbar(sc, ax=ax4, label="Sharpe", shrink=0.7)

    ax_tab = fig.add_axes([0.01, 0.01, 0.97, 0.12]); ax_tab.axis("off")
    cols_data = ["lookback", "entry_z", "exit_z", "sharpe",
                 "cagr", "max_dd", "win_rate", "n_trades", "profit_factor"]
    col_x_tab = np.linspace(0, 0.97, len(cols_data))
    headers = ["Lookback", "Entry Z", "Exit Z", "Sharpe",
               "CAGR%", "MaxDD%", "WinRate%", "Trades", "PF"]
    for j, (hdr, cx) in enumerate(zip(headers, col_x_tab)):
        ax_tab.text(cx, 0.92, hdr, fontsize=7.5, fontweight="bold",
                    color=YELLOW, transform=ax_tab.transAxes)
    for i, (_, row) in enumerate(optim_df.head(top_n).iterrows()):
        y = 0.82 - i * 0.075
        vals = [str(int(row.lookback)), str(row.entry_z), str(row.exit_z),
                f"{row.sharpe:.2f}", f"{row.cagr:+.2f}%", f"{row.max_dd:.2f}%",
                f"{row.win_rate:.1f}%", str(int(row.n_trades)), f"{row.profit_factor:.2f}"]
        col = YELLOW if i == 0 else WHITE
        for j, (val, cx) in enumerate(zip(vals, col_x_tab)):
            ax_tab.text(cx, y, val, fontsize=7, color=col,
                        transform=ax_tab.transAxes, fontfamily="monospace")

    plt.savefig("spx_page4_optimization.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("[OUTPUT] ✅ spx_page4_optimization.png")
    plt.close()