"""
training/trainer.py
===================
Boucle d'entraînement, validation et sauvegarde du modèle.

Responsabilités :
    - Entraîner le modèle epoch par epoch
    - Calculer la loss de validation à chaque epoch
    - Appliquer le scheduler ReduceLROnPlateau
    - Sauvegarder le meilleur checkpoint (meilleure val_loss)
    - Optionnel : early stopping
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """
    Effectue une epoch d'entraînement complète.

    Paramètres
    ----------
    model     : modèle PyTorch en mode train
    loader    : DataLoader d'entraînement
    criterion : fonction de loss
    optimizer : optimiseur
    device    : cpu ou cuda
    clip_grad : norme maximale pour le gradient clipping

    Retourne
    --------
    loss moyenne sur l'epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        if not torch.isfinite(loss):
            print("[WARN] Loss NaN/Inf détectée, batch ignoré.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Calcule la loss de validation sans mettre à jour les poids.

    Paramètres
    ----------
    model     : modèle PyTorch en mode eval
    loader    : DataLoader de validation
    criterion : fonction de loss
    device    : cpu ou cuda

    Retourne
    --------
    loss moyenne de validation
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)

def fit(
    model: nn.Module,
    loader_train: DataLoader,
    loader_val: DataLoader,
    device: torch.device,
    nb_epochs: int = 150,
    lr: float = 0.001,
    patience: int = 9999,
    checkpoint_path: str = "best_model.pth",
    log_every: int = 10,
) -> dict:
    """
    Entraîne le modèle et sauvegarde le meilleur checkpoint.

    Paramètres
    ----------
    model           : modèle CNN1D
    loader_train    : DataLoader d'entraînement
    loader_val      : DataLoader de validation
    device          : cpu ou cuda
    nb_epochs       : nombre maximum d'epochs
    lr              : learning rate initial d'Adam
    patience        : patience pour l'early stopping (9999 = désactivé)
    checkpoint_path : chemin de sauvegarde du meilleur modèle
    log_every       : affichage tous les N epochs

    Retourne
    --------
    history : dict avec les listes 'train_loss' et 'val_loss' par epoch
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_impr = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(nb_epochs):

        train_loss = train_one_epoch(
            model, loader_train, criterion, optimizer, device
        )
        val_loss = validate(model, loader_val, criterion, device)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_no_impr += 1
        if (epoch + 1) % log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:3d}/{nb_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {lr_now:.2e}"
            )
        if epochs_no_impr >= patience:
            print(f"\nEarly stopping à l'epoch {epoch+1} "
                  f"(pas d'amélioration depuis {patience} epochs).")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nMeilleure val_loss : {best_val_loss:.6f}")
        print(f"Checkpoint sauvegardé : {checkpoint_path}")

    return history
