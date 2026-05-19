"""
data_prep/prepare.py
====================
Nettoyage des CSV bruts et calcul des features d'entrée du modèle.

Entrée  : fichier CSV brut (format Yahoo Finance)
Sortie  : tableau numpy (N, 8) de features stationnarisées par log-diff

Features produites (toutes ancrées sur Close_HA) :
    0  log_ret   : ln(Close_HA[t] / Close_HA[t-1])
    1  log_open  : ln(Open_HA[t]  / Close_HA[t])
    2  log_high  : ln(High_HA[t]  / Close_HA[t])
    3  log_low   : ln(Low_HA[t]   / Close_HA[t])
    4  log_sma20 : ln(SMA20[t]    / Close_HA[t])
    5  log_sma50 : ln(SMA50[t]    / Close_HA[t])
    6  log_ema20 : ln(EMA20[t]    / Close_HA[t])
    7  log_ema50 : ln(EMA50[t]    / Close_HA[t])
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Fonctions internes
# ──────────────────────────────────────────────────────────────

def _load_csv(filepath: str) -> pd.DataFrame:
    """
    Charge un CSV Yahoo Finance, renomme la colonne Price en Date,
    supprime les doublons et trie par date.
    """
    df = pd.read_csv(filepath, skiprows=[1, 2])
    df = df.rename(columns={"Price": "Date"})
    df = df.drop_duplicates(subset=["Date"])
    df = df.sort_values(by="Date")

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprimer les lignes avec prix nul ou négatif (évite log(0))
    df = df[df["Close"] > 0].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.reset_index(drop=True)
    return df


def _compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les bougies Heikin-Ashi et les ajoute au DataFrame.
    Supprime les lignes où Close_HA <= 0.
    """
    df["Close_HA"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    open_ha = np.zeros(len(df))
    open_ha[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        open_ha[i] = (open_ha[i - 1] + df["Close_HA"].iloc[i - 1]) / 2

    df["Open_HA"] = open_ha
    df["High_HA"] = np.maximum(df["High"], np.maximum(df["Open_HA"], df["Close_HA"]))
    df["Low_HA"]  = np.minimum(df["Low"],  np.minimum(df["Open_HA"], df["Close_HA"]))

    df = df[df["Close_HA"] > 0].copy()
    df = df.reset_index(drop=True)
    return df


def _compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute SMA20, SMA50, EMA20, EMA50 calculées sur Close_HA.
    """
    df["SMA20"] = df["Close_HA"].rolling(window=20).mean()
    df["SMA50"] = df["Close_HA"].rolling(window=50).mean()
    df["EMA20"] = df["Close_HA"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close_HA"].ewm(span=50, adjust=False).mean()
    return df


def _log_anchor(series_vals: np.ndarray, close_vals: np.ndarray) -> np.ndarray:
    """
    Calcule ln(X[t] / Close_HA[t]).
    Retourne NaN pour toute valeur non positive.

    Paramètres
    ----------
    series_vals : valeurs de l'indicateur X, alignées avec close_vals
    close_vals  : valeurs de Close_HA servant d'ancrage
    """
    ratio = np.where(
        (series_vals > 0) & (close_vals > 0),
        series_vals / close_vals,
        np.nan,
    )
    return np.log(ratio)


def _build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Construit la matrice de features (N-1, 8) à partir du DataFrame
    nettoyé avec Heikin-Ashi et moyennes mobiles.

    Toutes les features sont des log-diffs stationnarisés :
    - log_ret   : variation temporelle de Close_HA
    - log_open/high/low/sma20/sma50/ema20/ema50 : ancrage sur Close_HA
    """
    c  = df["Close_HA"].values
    c1 = c[1:]   # Close_HA à t, pour aligner avec log_ret (qui est entre t-1 et t)

    # 1. Log-return : ln(C[t] / C[t-1])
    ratio   = c[1:] / c[:-1]
    ratio   = np.where(ratio > 0, ratio, np.nan)
    log_ret = np.log(ratio)

    # 2-4. OHLC Heikin-Ashi ancrés sur Close_HA
    log_open = _log_anchor(df["Open_HA"].values[1:], c1)
    log_high = _log_anchor(df["High_HA"].values[1:], c1)
    log_low  = _log_anchor(df["Low_HA"].values[1:],  c1)

    # 5-8. Moyennes mobiles ancrées sur Close_HA
    log_sma20 = _log_anchor(df["SMA20"].values[1:], c1)
    log_sma50 = _log_anchor(df["SMA50"].values[1:], c1)
    log_ema20 = _log_anchor(df["EMA20"].values[1:], c1)
    log_ema50 = _log_anchor(df["EMA50"].values[1:], c1)

    features = np.column_stack([
        log_ret, log_open, log_high, log_low,
        log_sma20, log_sma50, log_ema20, log_ema50,
    ])
    return features


def _remove_nan_rows(features: np.ndarray, filename: str) -> np.ndarray:
    """
    Supprime toute ligne contenant un NaN ou un Inf.
    Affiche un message si des lignes sont supprimées.
    """
    mask     = np.isfinite(features).all(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  [INFO] {n_dropped} ligne(s) supprimée(s) (NaN/Inf) dans {filename}")
    return features[mask]


# ──────────────────────────────────────────────────────────────
# Fonction publique principale
# ──────────────────────────────────────────────────────────────

def prepare_features(filepath: str) -> np.ndarray:
    """
    Pipeline complet de préparation des features pour un CSV brut.

    Paramètres
    ----------
    filepath : chemin vers le fichier CSV Yahoo Finance

    Retourne
    --------
    features : np.ndarray de forme (N, 8), dtype float64
               Chaque ligne est un jour, chaque colonne une feature
               stationnarisée par log-diff.
    """
    filename = os.path.basename(filepath)

    df = _load_csv(filepath)
    df = _compute_heikin_ashi(df)
    df = _compute_moving_averages(df)
    df = df.dropna().reset_index(drop=True)

    features = _build_features(df)
    features = _remove_nan_rows(features, filename)

    return features
