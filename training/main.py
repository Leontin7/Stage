"""
training/main.py
================
Point d'entrée principal. Orchestre les composantes :
    1. Chargement et préparation des features (data_prep)
    2. Construction des fenêtres et DataLoaders (dataset)
    3. Instanciation du modèle (model)
    4. Entraînement et sauvegarde (trainer)
    5. Évaluation sur le jeu de test

Usage
-----
    python training/main.py [csv_supplementaire_1.csv csv_supplementaire_2.csv ...]

Apple est toujours inclus dans le train. Le test se fait toujours sur Apple.

Exemple:
    python training/main.py Total.csv petrole.csv Umamusume.csv
"""

import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep.prepare  import prepare_features
from training.dataset   import make_windows, normalize, build_loaders
from training.model     import build_model
from training.trainer   import fit

CSV_DIR = "."
CSV_TARGET = "apple.csv"
CSV_FILES  = ["apple.csv"] + (sys.argv[1:] if len(sys.argv) > 1 else [])
WINDOW = 20
BATCH_SIZE = 50
NB_EPOCHS = 150
LR = 0.001
PATIENCE = 9999
CHECKPOINT = "best_model.pth"
all_train_x, all_train_y = [], []
all_val_x, all_val_y = [], []
features_target = prepare_features(os.path.join(CSV_DIR, CSV_TARGET))
n_target = len(features_target)
train_end = int(0.70 * n_target)
val_end = int(0.85 * n_target)
_, sigma_target = normalize(features_target[:train_end])

for csv_file in CSV_FILES:
    filepath = os.path.join(CSV_DIR, csv_file)
    print(f"Chargement: {csv_file}")
    features = prepare_features(filepath)
    n = len(features)
    t_end = int(0.70 * n)
    v_end = int(0.85 * n)
    train_f = features[:t_end]
    val_f = features[t_end:v_end]
    train_norm, val_norm, sigma_f = normalize(train_f, val_f)

    x_tr, y_tr = make_windows(train_norm, WINDOW)
    x_va, y_va = make_windows(val_norm, WINDOW)

    all_train_x.append(x_tr)
    all_train_y.append(y_tr)
    all_val_x.append(x_va)
    all_val_y.append(y_va)

    print(f"  -> train: {x_tr.shape}, val: {x_va.shape}")

x_train = np.concatenate(all_train_x, axis=0)
y_train = np.concatenate(all_train_y, axis=0)
x_val = np.concatenate(all_val_x, axis=0)
y_val = np.concatenate(all_val_y, axis=0)
print(f"\nDataset total — train: {x_train.shape}, val: {x_val.shape}")
test_f = features_target[val_end:]
test_norm = test_f / sigma_target
if not np.isfinite(test_norm).all():
    test_norm = np.nan_to_num(test_norm, nan=0.0, posinf=0.0, neginf=0.0)

x_test, y_test = make_windows(test_norm, WINDOW)
print(f"Test (Apple uniquement) : {x_test.shape}")
loader_train, loader_val, loader_test = build_loaders(
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    batch_size=BATCH_SIZE,
)
print(f"\nx_train: {x_train.shape} | Nb batches train: {len(loader_train)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")
model = build_model(n_features=8, dropout=0.2).to(device)
history = fit(
    model = model,
    loader_train = loader_train,
    loader_val = loader_val,
    device = device,
    nb_epochs = NB_EPOCHS,
    lr = LR,
    patience = PATIENCE,
    checkpoint_path = CHECKPOINT,
    log_every = 10,
)

def evaluate(
    model: torch.nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    csv_target: str,
) -> None:
    """
    Calcule et affiche les métriques sur le jeu de test.

    Paramètres
    ----------
    model      : modèle entraîné (meilleurs poids chargés)
    x_test     : features de test (N, window, F)
    y_test     : cibles de test (N,)
    device     : cpu ou cuda
    csv_target : nom du fichier testé (pour l'affichage)
    """
    x_tensor = torch.tensor(x_test.astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor).cpu().numpy()

    print(f"\nTest sur: {csv_target}")
    print("10 premières prédictions vs réelles :")
    print(f"{'Jour':>5}  {'Prédit':>10}  {'Réel':>10}  {'Erreur':>10}")
    print("-" * 42)
    for i in range(min(10, len(y_test))):
        erreur = abs(y_pred[i] - y_test[i])
        print(f"{i+1:>5}  {y_pred[i]:>10.4f}  {y_test[i]:>10.4f}  {erreur:>10.4f}")

    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    correct = np.sum(np.sign(y_pred) == np.sign(y_test))
    accuracy = correct / len(y_test) * 100

    print(f"\nMSE                  : {mse:.6f}")
    print(f"RMSE                 : {rmse:.6f}")
    print(f"Directional accuracy : {accuracy:.1f}%  ({correct}/{len(y_test)} jours)")


evaluate(model, x_test, y_test, device, CSV_TARGET)
