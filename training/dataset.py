"""
training/dataset.py
===================
Construction des fenêtres glissantes et DataLoaders PyTorch.

Chaque fenêtre produit deux cibles :
  y_dir  : hausse (1) / baisse (0) du log_ret MOYEN sur j+1..j+7
  y_bin  : bin du log_ret MOYEN sur j+1..j+7,
           calculé sur les valeurs brutes puis binnisé avec
           les edges globaux issus du fichier <nom>_edges.csv

Note : les tenseurs X sont en dtype=torch.long pour nn.Embedding.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_logret_edges(edges_csv_path: str) -> np.ndarray:
    """
    Lit un fichier <nom>_edges.csv généré par Binning.py.

    Format attendu : une colonne 'edge', n_bins+1 valeurs.
    Retourne un tableau 1-D (n_bins+1,) trié.
    """
    df = pd.read_csv(edges_csv_path)
    return df["edge"].values.astype(np.float64)


def logret_to_bin(logret_value: float, edges: np.ndarray) -> int:
    """
    Convertit une valeur brute de log_ret en bin_id (0..n_bins-1)
    en utilisant les mêmes edges que Binning.py (np.digitize sur les
    coupures intérieures).
    """
    inner = edges[1:-1]
    bin_id = int(np.digitize(logret_value, inner))
    return int(np.clip(bin_id, 0, len(edges) - 2))

def load_bin7_edges(bin7_csv_path: str) -> np.ndarray:
    """
    Lit un fichier <nom>_bin7_edges.csv généré par Binning.py.
    Ces edges sont calculés sur la distribution des MOYENNES de log_ret
    sur 7 jours — bien plus adaptés pour binner y_bin que les edges journaliers.
    """
    df = pd.read_csv(bin7_csv_path)
    return df["edge"].values.astype(np.float64)


def make_windows_from_binned(
    binned: np.ndarray,
    raw_logret: np.ndarray,
    logret_edges: np.ndarray,
    window: int,
    horizon_bin: int = 7,
    bin7_edges: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Découpe un tableau discrétisé en fenêtres glissantes avec deux cibles.

    Paramètres
    ----------
    binned        : (N, F) int64  — features discrétisées en bin_ids
    raw_logret    : (N,)   float  — valeurs brutes de log_ret
    logret_edges  : (n_bins+1,)   — edges journaliers (pour compatibilité)
    window        : taille de la fenêtre d'entrée
    horizon_bin   : nombre de jours futurs pour la moyenne (défaut 7)
    bin7_edges    : edges spécifiques pour la moyenne 7 jours (recommandé)
                    Si None, utilise logret_edges (moins précis)

    Retourne
    --------
    X      : (M, window, F) int64
    y_dir  : (M,)           int64  — 0=baisse, 1=hausse  (moyenne j+1..j+7)
    y_bin  : (M,)           int64  — bin 0..9 du log_ret moyen j+1..j+horizon
    """
    # Utilise bin7_edges si disponibles, sinon fallback sur logret_edges
    edges_for_bin = bin7_edges if bin7_edges is not None else logret_edges

    x_list, y_dir_list, y_bin_list = [], [], []

    max_i = len(binned) - window - horizon_bin
    for i in range(max_i):
        future_logrets = raw_logret[i + window : i + window + horizon_bin]
        if not np.all(np.isfinite(future_logrets)):
            continue
        mean_logret = float(np.mean(future_logrets))
        y_dir = 1 if mean_logret > 0 else 0
        y_bin = logret_to_bin(mean_logret, edges_for_bin)
        x_list.append(binned[i : i + window])
        y_dir_list.append(y_dir)
        y_bin_list.append(y_bin)

    return (
        np.array(x_list, dtype=np.int64),
        np.array(y_dir_list, dtype=np.int64),
        np.array(y_bin_list, dtype=np.int64),
    )

def make_forecast_windows(
    binned: np.ndarray,
    window: int,
    horizon: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fenêtres pour le pré-entraînement par forecasting auto-supervisé.

    Le modèle voit `window` jours et doit prédire les `horizon` jours suivants
    pour TOUTES les features — sans avoir besoin de labels humains.

    Paramètres
    ----------
    binned  : (N, F) int64  — features discrétisées
    window  : taille de la fenêtre d'entrée
    horizon : nombre de jours futurs à prédire (défaut 7)

    Retourne
    --------
    X_in    : (M, window, F)   int64  — fenêtre d'entrée
    X_fut   : (M, horizon, F)  int64  — jours futurs à prédire
    """
    x_in_list, x_fut_list = [], []
    max_i = len(binned) - window - horizon
    for i in range(max_i):
        x_in_list.append(binned[i : i + window])
        x_fut_list.append(binned[i + window : i + window + horizon])
    return (
        np.array(x_in_list,  dtype=np.int64),
        np.array(x_fut_list, dtype=np.int64),
    )

def build_forecast_loader(
    x_in: np.ndarray,
    x_fut: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader pour le pré-entraînement forecasting."""
    return DataLoader(
        TensorDataset(
            torch.tensor(x_in,  dtype=torch.long),
            torch.tensor(x_fut, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def apply_bins(
    features: np.ndarray,
    bin_edges_per_feature: list[np.ndarray],
) -> np.ndarray:
    N, F = features.shape
    binned = np.zeros((N, F), dtype=np.int64)
    for f in range(F):
        inner = bin_edges_per_feature[f][1:-1]
        binned[:, f] = np.digitize(features[:, f], inner).astype(np.int64)
    return binned

def build_loaders(
    x_train, y_dir_train, y_bin_train,
    x_val, y_dir_val, y_bin_val,
    x_test, y_dir_test, y_bin_test,
    batch_size: int = 50,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Chaque batch retourne : (x, y_dir, y_bin)
      x     : (batch, window, n_features) torch.long
      y_dir : (batch,) torch.long   — direction j+1
      y_bin : (batch,) torch.long   — bin moyen j+1..j+horizon

    Optimisations GPU :
      - pin_memory : mémoire hôte épinglée → transferts vers le GPU
        asynchrones (combiné à .to(device, non_blocking=True))
      - drop_last sur le train : tailles de batch constantes, ce qui permet
        à cudnn.benchmark de figer les kernels les plus rapides, et évite
        un dernier batch de taille 1 qui ferait planter BatchNorm
      - torch.from_numpy : partage la mémoire au lieu de copier
    """
    pin = torch.cuda.is_available()

    def _make_loader(x, y_dir, y_bin, shuffle, drop_last=False):
        return DataLoader(
            TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(y_dir),
                torch.from_numpy(y_bin),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin,
            drop_last=drop_last,
        )
    return (
        _make_loader(x_train, y_dir_train, y_bin_train, shuffle=True, drop_last=True),
        _make_loader(x_val, y_dir_val, y_bin_val, shuffle=False),
        _make_loader(x_test, y_dir_test, y_bin_test, shuffle=False),
    )
