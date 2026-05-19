"""
tools/visualize.py
==================
Outil de visualisation des données et des sorties du modèle.

Graphiques disponibles :
    1. plot_heikin_ashi()   — bougies HA + moyennes mobiles (SMA20/50, EMA20/50)
    2. plot_predictions()   — prédictions vs réelles sur le jeu de test
    3. plot_features()      — évolution des 8 features log-diff dans le temps
    4. plot_all()           — lance les 3 graphiques d'un coup

Usage
-----
    # Visualiser les bougies et indicateurs d'un CSV :
    python tools/visualize.py apple.csv

    # Visualiser aussi les prédictions d'un modèle sauvegardé :
    python tools/visualize.py apple.csv --model best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplfinance as mpf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.prepare import (
    _load_csv,
    _compute_heikin_ashi,
    _compute_moving_averages,
)

def plot_heikin_ashi(
    filepath: str,
    n_last: int = 200,
    save_path: str = None,
) -> None:
    """
    Affiche les bougies Heikin-Ashi avec SMA20, SMA50, EMA20, EMA50.

    Paramètres
    ----------
    filepath  : chemin vers le CSV brut
    n_last    : nombre de bougies à afficher (les plus récentes)
    save_path : si fourni, sauvegarde l'image au lieu de l'afficher
    """
    df = _load_csv(filepath)
    df = _compute_heikin_ashi(df)
    df = _compute_moving_averages(df)
    df = df.dropna().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.tail(n_last)
    ohlc_ha = pd.DataFrame({
        "Open":   df["Open_HA"],
        "High":   df["High_HA"],
        "Low":    df["Low_HA"],
        "Close":  df["Close_HA"],
    }, index=df.index)
    apds = [
        mpf.make_addplot(df["SMA20"], color="blue",   width=1.0, label="SMA20"),
        mpf.make_addplot(df["SMA50"], color="orange",  width=1.0, label="SMA50"),
        mpf.make_addplot(df["EMA20"], color="green",   width=0.8, linestyle="--", label="EMA20"),
        mpf.make_addplot(df["EMA50"], color="red",     width=0.8, linestyle="--", label="EMA50"),
    ]
    title = f"Heikin-Ashi — {os.path.basename(filepath)} ({n_last} dernières bougies)"
    kwargs = dict(
        type = "candle",
        addplot = apds,
        title = title,
        ylabel = "Prix",
        style = "charles",
        figsize = (16, 7),
        datetime_format = "%Y-%m",
        warn_too_much_data = n_last + 1,
    )
    if save_path:
        mpf.plot(ohlc_ha, **kwargs, savefig=save_path)
        print(f"Graphique sauvegardé : {save_path}")
    else:
        mpf.plot(ohlc_ha, **kwargs)
def plot_predictions(
    y_pred: np.ndarray,
    y_real: np.ndarray,
    title: str = "Prédictions vs Réelles",
    save_path: str = None,
) -> None:
    """
    Affiche trois sous-graphiques :
        - Haut   : prédictions et réelles superposées
        - Milieu : erreur absolue jour par jour
        - Bas    : direction correcte (vert) vs incorrecte (rouge)

    Paramètres
    ----------
    y_pred    : tableau (N,) des log-returns prédits
    y_real    : tableau (N,) des log-returns réels
    title     : titre du graphique
    save_path : si fourni, sauvegarde l'image
    """
    n = len(y_real)
    jours = np.arange(n)
    erreur = np.abs(y_pred - y_real)
    correct = np.sign(y_pred) == np.sign(y_real)
    accuracy = correct.mean() * 100

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{title}  —  Directional accuracy : {accuracy:.1f}%", fontsize=13)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1], hspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(jours, y_real, label="Réel", color="steelblue", linewidth=0.8, alpha=0.9)
    ax0.plot(jours, y_pred, label="Prédit", color="tomato", linewidth=0.8, alpha=0.9)
    ax0.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax0.set_ylabel("Log-return")
    ax0.legend(loc="upper right")
    ax0.set_title("Log-return prédit vs réel")
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.fill_between(jours, erreur, color="orange", alpha=0.6, label="Erreur absolue")
    ax1.set_ylabel("Erreur absolue")
    ax1.legend(loc="upper right")
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    colors = ["green" if c else "red" for c in correct]
    ax2.bar(jours, np.ones(n), color=colors, width=1.0, alpha=0.7)
    ax2.set_yticks([])
    ax2.set_xlabel("Jour")
    ax2.set_title("Direction : vert = correcte, rouge = incorrecte")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Graphique sauvegardé : {save_path}")
    else:
        plt.show()
def plot_features(
    features: np.ndarray,
    n_last: int = 500,
    save_path: str = None,
) -> None:
    """
    Affiche l'évolution temporelle des 8 features log-diff.
    Permet de vérifier visuellement la stationnarité et l'échelle.

    Paramètres
    ----------
    features  : tableau (N, 8) retourné par prepare_features()
    n_last    : nombre de pas de temps à afficher
    save_path : si fourni, sauvegarde l'image
    """
    noms = [
        "log_ret   (Close_HA t→t-1)",
        "log_open  (Open_HA / Close_HA)",
        "log_high  (High_HA / Close_HA)",
        "log_low   (Low_HA  / Close_HA)",
        "log_sma20 (SMA20   / Close_HA)",
        "log_sma50 (SMA50   / Close_HA)",
        "log_ema20 (EMA20   / Close_HA)",
        "log_ema50 (EMA50   / Close_HA)",
    ]

    data = features[-n_last:]
    t = np.arange(len(data))

    fig, axes = plt.subplots(8, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(f"Features log-diff — {n_last} derniers pas de temps", fontsize=13)

    for i, (ax, nom) in enumerate(zip(axes, noms)):
        ax.plot(t, data[:, i], linewidth=0.7, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(nom, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Pas de temps")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Graphique sauvegardé : {save_path}")
    else:
        plt.show()

def plot_all(
    filepath: str,
    y_pred: np.ndarray = None,
    y_real: np.ndarray = None,
    n_last_candles: int = 200,
    save_dir: str = None,
) -> None:
    """
    Lance les 3 graphiques en un seul appel.

    Paramètres
    ----------
    filepath       : chemin vers le CSV brut
    y_pred         : prédictions du modèle (optionnel)
    y_real         : valeurs réelles (optionnel)
    n_last_candles : nombre de bougies pour le graphique HA
    save_dir       : dossier de sauvegarde (None = affichage interactif)
    """
    from data_prep.prepare import prepare_features
    name = os.path.splitext(os.path.basename(filepath))[0]
    save1 = os.path.join(save_dir, f"{name}_heikin_ashi.png") if save_dir else None
    plot_heikin_ashi(filepath, n_last=n_last_candles, save_path=save1)
    features = prepare_features(filepath)
    save2 = os.path.join(save_dir, f"{name}_features.png") if save_dir else None
    plot_features(features, save_path=save2)
    if y_pred is not None and y_real is not None:
        save3 = os.path.join(save_dir, f"{name}_predictions.png") if save_dir else None
        plot_predictions(y_pred, y_real, title=f"Prédictions — {name}", save_path=save3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation des données et prédictions")
    parser.add_argument("csv", type=str, help="Fichier CSV à visualiser")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers best_model.pth")
    parser.add_argument("--save", type=str, default=None, help="Dossier de sauvegarde des images")
    parser.add_argument("--n", type=int, default=200, help="Nombre de bougies à afficher (défaut 200)")
    args = parser.parse_args()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
    from data_prep.prepare import prepare_features
    plot_heikin_ashi(args.csv, n_last=args.n, save_path=
        os.path.join(args.save, "heikin_ashi.png") if args.save else None)

    features = prepare_features(args.csv)
    plot_features(features, save_path=
        os.path.join(args.save, "features.png") if args.save else None)
    if args.model:
        import torch
        from training.dataset import make_windows, normalize
        from training.model   import build_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(n_features=8, dropout=0.2).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))

        n = len(features)
        val_end = int(0.85 * n)
        train_end = int(0.70 * n)

        _, sigma = normalize(features[:train_end])
        test_norm = features[val_end:] / sigma
        x_test, y_real = make_windows(test_norm, window=20)

        model.eval()
        with torch.no_grad():
            y_pred = model(
                torch.tensor(x_test.astype(np.float32)).to(device)
            ).cpu().numpy()

        plot_predictions(y_pred, y_real,
            title = f"Prédictions — {os.path.basename(args.csv)}",
            save_path = os.path.join(args.save, "predictions.png") if args.save else None,
        )
