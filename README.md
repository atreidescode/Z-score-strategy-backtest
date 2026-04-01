# Z-Score Mean Reversion Strategy — SPX/USD Backtest

Backtest d'une stratégie de **mean reversion** sur l'indice **S&P 500 (SPX/USD)** en timeframe **H1**, avec filtre de stationnarité ADF, signal Z-score et simulation réaliste des frais CFD.

---

## 📐 La Stratégie

### Principe
Les marchés financiers ont tendance à revenir vers leur moyenne après des écarts extrêmes. Cette stratégie exploite ce phénomène :

1. **Filtre ADF** — On vérifie que les rendements sont stationnaires sur une fenêtre glissante (`lookback_hours`). Si ce n'est pas le cas, on ne trade pas.
2. **Signal Z-score** — On calcule l'écart de la dernière bougie par rapport à la moyenne glissante, normalisé par l'écart-type.
3. **Entrée** — Quand le Z-score passe **sous le seuil d'entrée** (ex: -1.0), on achète (retour vers la moyenne attendu).
4. **Sortie** — Quand le Z-score remonte **au-dessus du seuil de sortie** (ex: +1.0), on clôture la position.

### Paramètres clés

| Paramètre | Défaut | Description |
|---|---|---|
| `lookback_hours` | 500 | Fenêtre glissante pour ADF et Z-score |
| `zscore_entry` | -1.0 | Seuil de Z-score pour entrer en position |
| `zscore_exit` | +1.0 | Seuil de Z-score pour sortir de position |
| `adf_pvalue` | 0.05 | Seuil de significativité du test ADF |
| `spread_points` | 0.5 | Spread CFD en points (simulation réaliste) |
| `swap_rate_annual` | 5% | Taux de financement overnight annualisé |
| `initial_capital` | $1,000,000 | Capital de départ |

---

## 📁 Structure du projet

```
Z-score-strategy-backtest/
│
├── data/
│   ├── convert_data.py          ← Script de conversion M1 → H1
│   ├── DAT_ASCII_SPXUSD_M1_XXXX.csv  ← Données brutes (non versionnées)
│   └── historique_3ans_H1.csv   ← Généré par convert_data.py (non versionné)
│
├── main.py                      ← Point d'entrée principal
├── config.py                    ← Tous les paramètres de la stratégie
├── data_loader.py               ← Chargement et validation des données
├── indicators.py                ← ADF, Z-score (fonctions pures)
├── backtest.py                  ← Moteur de backtest vectorisé
├── metrics.py                   ← Calcul des métriques de performance
├── optimizer.py                 ← Grid search sur les paramètres
└── plotting.py                  ← Génération des 4 pages de visualisations
```

---

## 📊 Télécharger les données

Les données M1 du SPX/USD sont disponibles gratuitement sur **HistData** :

1. Aller sur [https://www.histdata.com/download-free-forex-data/](https://www.histdata.com/download-free-forex-data/)
2. Choisir **Indices → SPX500** (ou chercher `SPXUSD`)
3. Sélectionner le format **ASCII** et les années souhaitées (ex: 2023, 2024, 2025)
4. Télécharger et décompresser les fichiers dans le dossier `data/`

Les fichiers doivent être nommés :
```
DAT_ASCII_SPXUSD_M1_2023.csv
DAT_ASCII_SPXUSD_M1_2024.csv
DAT_ASCII_SPXUSD_M1_2025.csv
```

Format attendu (séparateur `;`, sans header) :
```
20240101 180000;4770.867000;4773.177000;4769.734000;4772.677000;0
```

---

## ⚙️ Installation

```bash
pip install pandas numpy statsmodels matplotlib seaborn
```

---

## 🚀 Exécution

### Étape 1 — Conversion des données M1 → H1
```bash
cd data/
python convert_data.py
```
Génère `data/historique_3ans_H1.csv`.

### Étape 2 — Lancer le backtest
```bash
python main.py
```

### Résultats générés
4 fichiers PNG dans le dossier racine :
- `spx_page1_overview.png` — Vue d'ensemble et tableau comparatif
- `spx_page2_performance.png` — Analyse de performance détaillée
- `spx_page3_trades.png` — Analyse microscopique des trades
- `spx_page4_optimization.png` — Heatmaps du grid search

---

## 🔧 Configuration

Tous les paramètres sont dans `config.py`. Pour modifier la stratégie :

```python
@dataclass
class Config:
    lookback_hours: int = 500       # Fenêtre glissante
    zscore_entry: float = -1.0      # Seuil d'entrée
    zscore_exit: float = 1.0        # Seuil de sortie
    optimize: bool = True           # Activer le grid search
```

Pour désactiver l'optimisation (plus rapide) :
```python
optimize: bool = False
```

---

## ⚠️ Disclaimer

Ce projet est à but **éducatif uniquement**. Les performances passées ne garantissent pas les performances futures. Ne pas utiliser en trading réel sans validation approfondie.
