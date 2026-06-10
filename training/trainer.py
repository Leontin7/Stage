"""
training/trainer.py
===================
Boucle d'entraînement pour la double tête de prédiction :
  - Tête 1 (direction) : CrossEntropyLoss sur 2 classes
  - Tête 2 (bin moyen) : CrossEntropyLoss sur n_bins classes

La loss totale est une somme pondérée :
    loss = w_dir * loss_dir + w_bin * loss_bin
"""

import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.utils.data import DataLoader

W_DIR_DEFAULT = 1.0
W_BIN_DEFAULT = 0.5


def train_one_epoch(model, loader, criterion, optimizer, device,
                    w_dir=W_DIR_DEFAULT, w_bin=W_BIN_DEFAULT,
                    clip_grad=1.0, scaler=None, use_amp=False):
    model.train()
    total_loss, n_batches = 0.0, 0

    for x_batch, y_dir_batch, y_bin_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_dir_batch = y_dir_batch.to(device, non_blocking=True)
        y_bin_batch = y_bin_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda") if use_amp else nullcontext():
            logits_dir, logits_bin = model(x_batch)
            loss_dir = criterion(logits_dir, y_dir_batch)
            loss_bin = criterion(logits_bin, y_bin_batch)
            loss = w_dir * loss_dir + w_bin * loss_bin

        if not torch.isfinite(loss):
            print("[WARN] Loss NaN/Inf, batch ignoré.")
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device,
             w_dir=W_DIR_DEFAULT, w_bin=W_BIN_DEFAULT, use_amp=False):
    model.eval()
    total_loss, correct_dir, correct_bin, total = 0.0, 0, 0, 0
    with torch.inference_mode():
        for x_batch, y_dir_batch, y_bin_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_dir_batch = y_dir_batch.to(device, non_blocking=True)
            y_bin_batch = y_bin_batch.to(device, non_blocking=True)

            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                logits_dir, logits_bin = model(x_batch)
                loss_dir = criterion(logits_dir, y_dir_batch)
                loss_bin = criterion(logits_bin, y_bin_batch)
                loss = w_dir * loss_dir + w_bin * loss_bin

            total_loss += loss.item()
            correct_dir += (logits_dir.argmax(dim=1) == y_dir_batch).sum().item()
            correct_bin += (logits_bin.argmax(dim=1) == y_bin_batch).sum().item()
            total += len(y_dir_batch)

    acc_dir = correct_dir / max(total, 1) * 100
    acc_bin = correct_bin / max(total, 1) * 100
    return total_loss / max(len(loader), 1), acc_dir, acc_bin


def fit(model, loader_train, loader_val, device,
        nb_epochs=150, lr=0.001, patience=9999,
        w_dir=W_DIR_DEFAULT, w_bin=W_BIN_DEFAULT,
        checkpoint_path="best_model.pth", log_every=10, use_amp=None):
    use_amp = (device.type == "cuda") if use_amp is None else use_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    base_model = getattr(model, "_orig_mod", model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    best_val_loss = float("inf")
    best_state = None
    epochs_no_impr = 0
    history = {"train_loss": [], "val_loss": [],
                      "val_acc_dir": [], "val_acc_bin": []}
    for epoch in range(nb_epochs):
        train_loss = train_one_epoch(
            model, loader_train, criterion, optimizer, device,
            w_dir=w_dir, w_bin=w_bin, scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_acc_dir, val_acc_bin = validate(
            model, loader_val, criterion, device,
            w_dir=w_dir, w_bin=w_bin, use_amp=use_amp,
        )

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc_dir"].append(val_acc_dir)
        history["val_acc_bin"].append(val_acc_bin)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in base_model.state_dict().items()}
            epochs_no_impr = 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_no_impr += 1

        if (epoch + 1) % log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d}/{nb_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc dir: {val_acc_dir:.1f}% | "
                  f"Val Acc bin: {val_acc_bin:.1f}% | "
                  f"LR: {lr_now:.2e}")

        if epochs_no_impr >= patience:
            print(f"\nEarly stopping à l'epoch {epoch+1}.")
            break

    if best_state is not None:
        base_model.load_state_dict(best_state)
        print(f"\nMeilleure val_loss : {best_val_loss:.6f}")
        print(f"Checkpoint sauvegardé : {checkpoint_path}")

    return history
