"""
config.py — Configuration centrale de la stratégie SPX Mean Reversion.
Tous les paramètres sont regroupés dans un dataclass immuable.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Données ────────────────────────────────────────────────────────────────
    data_path: str = "../data/historique_3ans_H1.csv"  # ← seule ligne modifiée

    # ── Paramètres de la stratégie ─────────────────────────────────────────────
    lookback_hours: int = 500
    zscore_entry: float = -1.0
    zscore_exit: float = 1.0
    adf_pvalue: float = 0.05
    market_open_hour: int = 9
    market_open_minute: int = 30

    # ── Capital & frais ────────────────────────────────────────────────────────
    initial_capital: float = 1_000_000.0
    spread_points: float = 0.5
    swap_rate_annual: float = 0.05
    commission_per_trade: float = 0.0

    # ── Optimisation ──────────────────────────────────────────────────────────
    optimize: bool = True
    top_n_combinations: int = 5
    lookback_range: List[int] = field(default_factory=lambda: [300, 400, 500, 600])
    zscore_entry_range: List[float] = field(default_factory=lambda: [-0.5, -1.0, -1.5, -2.0])
    zscore_exit_range: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Compatibilité avec l'ancien format CONFIG dict."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)