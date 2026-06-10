"""
tools/Binning.py
================
Découpage en bins équipeuplés (même nombre d'éléments par bin)
sur TOUTES les features issues de prepare_features().

Nombre de bins par feature (configurable via N_BINS_PER_FEATURE) :
    log_ret    : 10
    log_mm20   :  3
    log_mm50   :  3
    bb_haute   :  3
    bb_basse   :  3
    rsi14      :  3
    macd_line  :  2
    macd_signal:  2

Sortie
------
    - data/<nom>_bin.csv    — bin_ids par feature, une ligne par date
    - data/<nom>_edges.csv  — edges de log_ret (calculés sur distribution globale)

Usage
-----
    python tools/Binning.py                        # tous les CSV de data/
    python tools/Binning.py data/apple.csv ...     # fichiers spécifiques
"""

import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep.prepare import prepare_features_with_dates

FEATURE_NAMES = ["log_ret", "log_mm20", "log_mm50", "bb_haute", "bb_basse",
                 "rsi14", "macd_line", "macd_signal"]
N_BINS_PER_FEATURE = [10, 3, 3, 3, 3, 3, 2, 2]

def bin_feature(values: np.ndarray, feature_name: str, n_bins: int) -> pd.DataFrame:
    cut, bins_edges = pd.qcut(
        values, q=n_bins, labels=False, retbins=True, duplicates="drop",
    )
    n_eff = len(bins_edges) - 1
    if n_eff != n_bins:
        print(f"  [WARN] {feature_name} : {n_eff} bins effectifs au lieu de "
              f"{n_bins} (edges dupliqués) — vérifie N_BINS_PER_FEATURE.")
    df = pd.DataFrame({"value": values, "bin": cut.astype(int)})
    n_total = len(df)
    rows = []
    for bin_id in sorted(df["bin"].unique()):
        mask   = df["bin"] == bin_id
        subset = df.loc[mask, "value"]
        rows.append({
            "feature": feature_name,
            "bin_id": bin_id,
            "n_bins": n_bins,
            "left_edge": bins_edges[bin_id],
            "right_edge": bins_edges[bin_id + 1],
            "count":len(subset),
            "pct": round(len(subset) / n_total * 100, 2),
            "mean": round(subset.mean(), 6),
            "std": round(subset.std(), 6),
            "min": round(subset.min(), 6),
            "max": round(subset.max(), 6),
        })
    return pd.DataFrame(rows)


def print_summary(df_bins: pd.DataFrame) -> None:
    for feature in df_bins["feature"].unique():
        sub = df_bins[df_bins["feature"] == feature].copy()
        n_bins = sub["n_bins"].iloc[0]
        print(f"\n{'═' * 72}")
        print(f"  Feature : {feature}   ({n_bins} bins)")
        print(f"{'═' * 72}")
        print(f"{'Bin':>4}  {'Left':>12}  {'Right':>12}  {'Count':>7}  {'%':>6}  {'Mean':>10}  {'Std':>10}")
        print(f"{'-' * 72}")
        for _, row in sub.iterrows():
            print(
                f"{int(row['bin_id']):>4}  "
                f"{row['left_edge']:>12.6f}  {row['right_edge']:>12.6f}  "
                f"{int(row['count']):>7}  {row['pct']:>5.1f}%  "
                f"{row['mean']:>10.6f}  {row['std']:>10.6f}"
            )


def compute_bin7_edges(
    all_features: list[np.ndarray],
    window: int = 20,
    horizon: int = 7,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Calcule les edges équipeuplés sur la distribution des MOYENNES de log_ret
    sur `horizon` jours consécutifs — à utiliser pour binner y_bin.

    C'est une distribution bien plus étroite que le log_ret journalier brut,
    donc les edges journaliers ne conviennent pas pour y_bin.
    """
    all_means = []
    for feat in all_features:
        logret = feat[:, 0]
        n = len(logret)
        for i in range(n - horizon):
            chunk = logret[i : i + horizon]
            if np.all(np.isfinite(chunk)):
                all_means.append(float(np.mean(chunk)))
    all_means = np.array(all_means)
    _, edges = pd.qcut(all_means, q=n_bins, labels=False, retbins=True, duplicates="drop")
    edges[0] = -np.inf
    edges[-1] =  np.inf
    return edges

def save_bin7_edges(edges: np.ndarray, out_path: str) -> None:
    pd.DataFrame({"edge": edges}).to_csv(out_path, index=False)
    print(f"  ✓ Edges bin7 sauvegardés : {out_path}")

def save_logret_edges(logret_edges: np.ndarray, out_path: str) -> None:
    """
    Sauvegarde les edges de log_ret dans un CSV léger.
    Format : une colonne 'edge', n_bins_logret+1 valeurs.
    """
    pd.DataFrame({"edge": logret_edges}).to_csv(out_path, index=False)
    print(f"  ✓ Edges sauvegardés : {out_path}")

def save_binned_csv(
    features: np.ndarray,
    dates: np.ndarray,
    bin_edges_per_feature: list,
    n_bins_per_feature: list,
    out_path: str,
) -> None:
    """
    Discrétise les features avec les bin_edges fournis et sauvegarde le CSV.
    Chaque feature est clippée à [0, n_bins-1] après np.digitize.
    """
    N, F = features.shape
    binned = np.zeros((N, F), dtype=np.int64)
    for f in range(F):
        inner = bin_edges_per_feature[f][1:-1]
        raw = np.digitize(features[:, f], inner).astype(np.int64)
        binned[:, f] = np.clip(raw, 0, n_bins_per_feature[f] - 1)
    df_out = pd.DataFrame(binned, columns=FEATURE_NAMES)
    df_out.insert(0, "Date", dates)
    df_out.to_csv(out_path, index=False)
    print(f"  ✓ Sauvegardé : {out_path}  ({N} lignes)")

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    else:
        csv_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".csv")
            and not f.endswith("_bin.csv")
            and not f.endswith("_edges.csv")
        )
        print(f"Aucun argument — chargement de tous les CSV de {data_dir}/\n")
    all_features, all_dates, valid_paths = [], [], []

    for path in csv_files:
        if not os.path.isabs(path):
            candidate = os.path.join(data_dir, os.path.basename(path))
            path = candidate if os.path.exists(candidate) else path
        print(f"Chargement : {path}")
        try:
            feat, dates = prepare_features_with_dates(path)
            print(f"  -> {feat.shape[0]} lignes")
            all_features.append(feat)
            all_dates.append(dates)
            valid_paths.append(path)
        except Exception as e:
            print(f"  [ERREUR] {e}")

    if not all_features:
        print("Aucune donnée chargée, abandon.")
        return

    features_global = np.concatenate(all_features, axis=0)
    print(f"\nDataset global : {features_global.shape[0]} observations\n")
    print("Bins par feature :")
    for name, nb in zip(FEATURE_NAMES, N_BINS_PER_FEATURE):
        print(f"  {name:<14} : {nb} bins")
    print()
    bin_edges_per_feature = []
    all_bins_rows = []
    for i, (name, n_bins) in enumerate(zip(FEATURE_NAMES, N_BINS_PER_FEATURE)):
        vals = features_global[:, i]
        vals_clean = vals[np.isfinite(vals)]
        df_b = bin_feature(vals_clean, name, n_bins)
        all_bins_rows.append(df_b)
        edges = df_b["left_edge"].tolist() + [df_b["right_edge"].iloc[-1]]
        bin_edges_per_feature.append(np.array(edges))
    df_all = pd.concat(all_bins_rows, ignore_index=True)
    print_summary(df_all)
    logret_edges = bin_edges_per_feature[0]

    print("\nGénération des fichiers _bin.csv, _edges.csv et _bin7_edges.csv :")
    for path, feat, dates in zip(valid_paths, all_features, all_dates):
        basename = os.path.splitext(os.path.basename(path))[0]
        bin_path = os.path.join(data_dir, f"{basename}_bin.csv")
        edges_path = os.path.join(data_dir, f"{basename}_edges.csv")
        bin7_path = os.path.join(data_dir, f"{basename}_bin7_edges.csv")
        save_binned_csv(feat, dates, bin_edges_per_feature, N_BINS_PER_FEATURE, bin_path)
        save_logret_edges(logret_edges, edges_path)
        bin7_edges_stock = compute_bin7_edges([feat], horizon=7, n_bins=10)
        save_bin7_edges(bin7_edges_stock, bin7_path)

if __name__ == "__main__":
    main()
