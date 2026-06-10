"""
training/bert_mlm.py
====================
Modèle BERT-style pré-entraîné sur la prédiction du bin moyen de log_ret
sur les 7 prochains jours — exactement la même tâche que le fine-tuning.

Principe
--------
  Pré-entraînement : fenêtre 20 jours → prédire y_bin  (bin moyen log_ret j+1..j+7)
  Fine-tuning      : fenêtre 20 jours → prédire y_dir + y_bin

Les deux tâches sont alignées : le pré-entraînement apprend directement
les patterns qui précèdent une hausse ou baisse sur 7 jours.

Architecture
------------
  1. Embedding par feature (learnable, un par feature)
  2. Token [CLS] learnable
  3. Positional encoding learnable
  4. N couches TransformerEncoder (Pre-LN)
  5. Tête bin  : [CLS] → bin moyen (pré-entraînement + fine-tuning)
  6. Tête dir  : [CLS] → direction (fine-tuning uniquement)
"""

from __future__ import annotations
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BertBinMLM(nn.Module):

    def __init__(
        self,
        n_bins_per_feature: list[int],
        emb_dim: int = 8,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        n_bins_logret: int = 10,
    ):
        super().__init__()

        self.n_features = len(n_bins_per_feature)
        self.n_bins_per_feature = n_bins_per_feature
        self.emb_dim = emb_dim
        self.d_model = self.n_features * emb_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_bins, emb_dim)
            for n_bins in n_bins_per_feature
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_emb = nn.Embedding(max_len + 1, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.fc_shared = nn.Linear(self.d_model, 128)
        self.head_bin = nn.Linear(128, n_bins_logret)
        self.head_direction = nn.Linear(128, 2)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        embs = [self.embeddings[f](x[:, :, f]) for f in range(F)]
        tok_emb = torch.cat(embs, dim=-1)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok_emb], dim=1)
        positions = torch.arange(1 + T, device=x.device).unsqueeze(0)
        seq = seq + self.pos_emb(positions)
        return self.transformer(seq)

    def forward(self, x: torch.Tensor):
        """Retourne (logits_dir, logits_bin) — interface identique à CNN1D."""
        out = self._encode(x)
        cls_repr = out[:, 0, :]
        shared = self.act(self.fc_shared(self.drop(cls_repr)))
        return self.head_direction(shared), self.head_bin(shared)

    def forward_pretrain(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne logits_bin uniquement — utilisé en pré-entraînement."""
        out = self._encode(x)
        cls_repr = out[:, 0, :]
        shared = self.act(self.fc_shared(self.drop(cls_repr)))
        return self.head_bin(shared)


def build_bert_model(
    n_bins_per_feature: list[int],
    emb_dim: int = 8,
    n_heads: int = 4,
    n_layers: int = 3,
    ff_dim: int = 256,
    dropout: float = 0.1,
) -> BertBinMLM:
    model = BertBinMLM(
        n_bins_per_feature=n_bins_per_feature,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        n_bins_logret=n_bins_per_feature[0],
    )
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle BertBinMLM instancié — {nb_params:,} paramètres entraînables")
    print(f"  d_model={model.d_model} | n_heads={n_heads} | n_layers={n_layers} | ff_dim={ff_dim}")
    print(f"  bins/feature : {n_bins_per_feature}")
    print(f"  Sorties : dir(2) + bin({n_bins_per_feature[0]})")
    return model

def fit_mlm(
    model: BertBinMLM,
    loader_train: DataLoader,
    loader_val: DataLoader,
    n_bins_per_feature: list[int],
    device: torch.device,
    nb_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    log_every: int = 5,
    checkpoint_path: str = "bert_pretrain.pth",
    **kwargs,
) -> dict:
    """
    Pré-entraînement : le modèle prédit y_bin (bin moyen log_ret j+1..j+7)
    depuis la fenêtre de 20 jours — exactement la même tâche que le fine-tuning.
    Loader : (x, y_dir, y_bin) — on utilise seulement x et y_bin.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    base_model = getattr(model, "_orig_mod", model)
    best_val_loss, best_state, epochs_no_impr = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    print("\n" + "═" * 60)
    print("  Pré-entraînement BERT — Prédiction bin moyen 7 jours")
    print("═" * 60)
    print(f"  Fenêtres train : {len(loader_train.dataset):,}  |  val : {len(loader_val.dataset):,}")
    print(f"  Tâche : fenêtre 20j → bin moyen log_ret j+1..j+7")
    print(f"  Epochs : {nb_epochs}  |  LR : {lr}\n")

    for epoch in range(nb_epochs):
        model.train()
        total_loss, correct, total, n_batches = 0.0, 0, 0, 0
        for x_batch, _, y_bin_batch in loader_train:
            x_batch = x_batch.to(device, non_blocking=True)
            y_bin_batch = y_bin_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                logits = model.forward_pretrain(x_batch)
                loss = criterion(logits, y_bin_batch)
            if not torch.isfinite(loss):
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y_bin_batch).sum().item()
            total += len(y_bin_batch)
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.inference_mode():
            for x_batch, _, y_bin_batch in loader_val:
                x_batch = x_batch.to(device, non_blocking=True)
                y_bin_batch = y_bin_batch.to(device, non_blocking=True)
                with torch.amp.autocast("cuda") if use_amp else nullcontext():
                    logits = model.forward_pretrain(x_batch)
                    loss_val = criterion(logits, y_bin_batch)
                val_loss += loss_val.item()
                val_correct += (logits.argmax(1) == y_bin_batch).sum().item()
                val_total += len(y_bin_batch)
        val_loss /= max(len(loader_val), 1)
        val_acc = val_correct / max(val_total, 1) * 100
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

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
            print(f"  Epoch {epoch+1:3d}/{nb_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Acc bin: {val_acc:.1f}% | LR: {lr_now:.2e}")

        if epochs_no_impr >= patience:
            print(f"\n  Early stopping à l'epoch {epoch+1}.")
            break

    if best_state is not None:
        base_model.load_state_dict(best_state)
        print(f"\n  Meilleure val_loss pré-entraînement : {best_val_loss:.6f}")
        print(f"  Checkpoint : {checkpoint_path}")

    return history

def fit_finetune(
    model: BertBinMLM,
    loader_train: DataLoader,
    loader_val: DataLoader,
    device: torch.device,
    nb_epochs: int = 150,
    lr: float = 2e-4,
    patience: int = 20,
    w_dir: float = 1.0,
    w_bin: float = 0.5,
    checkpoint_path: str = "bert_finetune.pth",
    log_every: int = 10,
    **kwargs,
) -> dict:
    """
    Fine-tuning : tout le modèle est entraîné dès le début avec un LR faible.
    Pas de gel progressif — les représentations du pré-entraînement sont déjà
    alignées avec la tâche (même prédiction de y_bin).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    base_model = getattr(model, "_orig_mod", model)

    best_val_loss, best_state, epochs_no_impr = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc_dir": [], "val_acc_bin": []}

    print("\n" + "═" * 60)
    print("  Fine-tuning BERT — tout dégelé, LR faible")
    print("═" * 60)
    print(f"  Epochs : {nb_epochs}  |  LR : {lr}  |  W_dir : {w_dir}  |  W_bin : {w_bin}\n")

    for epoch in range(nb_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x_batch, y_dir_batch, y_bin_batch in loader_train:
            x_batch = x_batch.to(device, non_blocking=True)
            y_dir_batch = y_dir_batch.to(device, non_blocking=True)
            y_bin_batch = y_bin_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                logits_dir, logits_bin = model(x_batch)
                loss = (w_dir * criterion(logits_dir, y_dir_batch) +
                        w_bin * criterion(logits_bin, y_bin_batch))
            if not torch.isfinite(loss):
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss, correct_dir, correct_bin, total = 0.0, 0, 0, 0
        with torch.inference_mode():
            for x_batch, y_dir_batch, y_bin_batch in loader_val:
                x_batch = x_batch.to(device, non_blocking=True)
                y_dir_batch = y_dir_batch.to(device, non_blocking=True)
                y_bin_batch = y_bin_batch.to(device, non_blocking=True)
                with torch.amp.autocast("cuda") if use_amp else nullcontext():
                    logits_dir, logits_bin = model(x_batch)
                    loss = (w_dir * criterion(logits_dir, y_dir_batch) +
                            w_bin * criterion(logits_bin, y_bin_batch))
                val_loss += loss.item()
                correct_dir += (logits_dir.argmax(1) == y_dir_batch).sum().item()
                correct_bin += (logits_bin.argmax(1) == y_bin_batch).sum().item()
                total += len(y_dir_batch)
        val_loss /= max(len(loader_val), 1)
        acc_dir = correct_dir / max(total, 1) * 100
        acc_bin = correct_bin / max(total, 1) * 100
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc_dir"].append(acc_dir)
        history["val_acc_bin"].append(acc_bin)

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
            print(f"  Epoch {epoch+1:3d}/{nb_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Dir: {acc_dir:.1f}% | Bin: {acc_bin:.1f}% | "
                  f"LR: {lr_now:.2e}")

        if epochs_no_impr >= patience:
            print(f"\n  Early stopping à l'epoch {epoch+1}.")
            break

    if best_state is not None:
        base_model.load_state_dict(best_state)
        print(f"\n  Meilleure val_loss fine-tune : {best_val_loss:.6f}")
        print(f"  Checkpoint : {checkpoint_path}")

    return history
