"""
data_prep/prepare.py
====================
Nettoyage des CSV bruts et calcul des features d'entrée du modèle.

Entrée  : fichier CSV brut (format Yahoo Finance)
Sortie  : tableau numpy (N, 14) de features stationnarisées

Features produites :
    0  log_ret    : ln(Close_HA[t] / Close_HA[t-1])
    1  log_mm20   : ln(SMA20[t]    / Close_HA[t])
    2  log_mm50   : ln(SMA50[t]    / Close_HA[t])
    3  bb_haute   : ln(BB_haute[t] / Close_HA[t])
    4  bb_basse   : ln(BB_basse[t] / Close_HA[t])
    5  rsi14      : RSI 14 périodes / 100  ∈ [0,1]
    6  macd_line  : (EMA12 - EMA26) / Close_HA
    7  macd_signal: EMA9(macd_line) / Close_HA
"""

import os
import numpy as np
import pandas as pd


def _load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, skiprows=[1, 2])
    df = df.rename(columns={"Price": "Date"})
    df = df.drop_duplicates(subset=["Date"])
    df = df.sort_values(by="Date")
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Close"] > 0].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.reset_index(drop=True)
    return df


def _compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df["Close_HA"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    close_ha = df["Close_HA"].to_numpy()
    open_ha = np.empty(len(df))
    open_ha[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        open_ha[i] = (open_ha[i - 1] + close_ha[i - 1]) / 2
    df["Open_HA"] = open_ha
    df["High_HA"] = np.maximum(df["High"], np.maximum(df["Open_HA"], df["Close_HA"]))
    df["Low_HA"] = np.minimum(df["Low"],  np.minimum(df["Open_HA"], df["Close_HA"]))
    df = df[df["Close_HA"] > 0].copy()
    return df.reset_index(drop=True)

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close_HA"]

    df["MM20"] = c.rolling(window=20).mean()
    df["MM50"] = c.rolling(window=50).mean()

    std20 = c.rolling(window=20).std()
    df["BB_haute"] = df["MM20"] + 2 * std20
    df["BB_basse"] = df["MM20"] - 2 * std20
    denom = (df["BB_haute"] - df["BB_basse"]).replace(0, np.nan)
    df["BB_pct"] = (c - df["BB_basse"]) / denom   # ∈ [0, 1]

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD_line"] = ema12 - ema26
    df["MACD_signal"]= df["MACD_line"].ewm(span=9, adjust=False).mean()

    return df


def _log_anchor(series_vals: np.ndarray, close_vals: np.ndarray) -> np.ndarray:
    ratio = np.where(
        (series_vals > 0) & (close_vals > 0),
        series_vals / close_vals,
        np.nan,
    )
    return np.log(ratio)


def _build_features(df: pd.DataFrame) -> np.ndarray:
    c  = df["Close_HA"].values
    c1 = c[1:]

    ratio = c[1:] / c[:-1]
    log_ret = np.log(np.where(ratio > 0, ratio, np.nan))

    log_mm20 = _log_anchor(df["MM20"].values[1:], c1)
    log_mm50 = _log_anchor(df["MM50"].values[1:], c1)

    bb_haute = _log_anchor(df["BB_haute"].values[1:], c1)
    bb_basse = _log_anchor(df["BB_basse"].values[1:], c1)
    rsi14 = df["RSI14"].values[1:] / 100.0

    macd_line = df["MACD_line"].values[1:]   / c1
    macd_signal = df["MACD_signal"].values[1:] / c1

    features = np.column_stack([
        log_ret,
        log_mm20, log_mm50,
        bb_haute, bb_basse,
        rsi14,
        macd_line, macd_signal,
    ])
    return features


def _remove_nan_rows(features: np.ndarray, filename: str) -> np.ndarray:
    mask = np.isfinite(features).all(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  [INFO] {n_dropped} ligne(s) supprimée(s) (NaN/Inf) dans {filename}")
    return features[mask]


def _remove_nan_rows_with_dates(
    features: np.ndarray, dates: np.ndarray, filename: str
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(features).all(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  [INFO] {n_dropped} ligne(s) supprimée(s) (NaN/Inf) dans {filename}")
    return features[mask], dates[mask]


def _build_features_with_dates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Même chose que _build_features mais retourne aussi les dates alignées."""
    c  = df["Close_HA"].values
    c1 = c[1:]

    ratio   = c[1:] / c[:-1]
    log_ret = np.log(np.where(ratio > 0, ratio, np.nan))

    log_mm20 = _log_anchor(df["MM20"].values[1:],        c1)
    log_mm50 = _log_anchor(df["MM50"].values[1:],        c1)
    bb_haute = _log_anchor(df["BB_haute"].values[1:],    c1)
    bb_basse = _log_anchor(df["BB_basse"].values[1:],    c1)
    rsi14 = df["RSI14"].values[1:] / 100.0
    macd_line = df["MACD_line"].values[1:]   / c1
    macd_signal = df["MACD_signal"].values[1:] / c1

    features = np.column_stack([
        log_ret, log_mm20, log_mm50,
        bb_haute, bb_basse,
        rsi14, macd_line, macd_signal,
    ])
    dates = df["Date"].values[1:]
    return features, dates

def prepare_features(filepath: str) -> np.ndarray:
    """
    Pipeline complet de préparation des features pour un CSV brut.

    Retourne
    --------
    features : np.ndarray de forme (N, 8), dtype float64
    """
    filename = os.path.basename(filepath)
    df = _load_csv(filepath)
    df = _compute_heikin_ashi(df)
    df = _compute_indicators(df)
    df = df.dropna().reset_index(drop=True)
    features = _build_features(df)
    features = _remove_nan_rows(features, filename)
    return features


def prepare_features_with_dates(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Même pipeline que prepare_features, mais retourne aussi les dates.

    Retourne
    --------
    features : np.ndarray de forme (N, 8), dtype float64
    dates    : np.ndarray de forme (N,),   dtype str
    """
    filename = os.path.basename(filepath)
    df = _load_csv(filepath)
    df = _compute_heikin_ashi(df)
    df = _compute_indicators(df)
    df = df.dropna().reset_index(drop=True)
    features, dates = _build_features_with_dates(df)
    features, dates = _remove_nan_rows_with_dates(features, dates, filename)
    return features, dates
