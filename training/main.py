"""
training/main.py
================
Point d'entrée principal — double tête de prédiction.

Usage
-----
    python training/main.py                  # charge tous les _bin.csv de data/
    python training/main.py apple_bin.csv    # force un fichier test spécifique
"""

import os
import sys
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset import make_windows_from_binned, build_loaders, load_logret_edges, load_bin7_edges
from training.model   import build_model
from training.trainer import fit

WINDOW = 20
BATCH_SIZE = 50
NB_EPOCHS = 150
LR = 0.001
PATIENCE = 15
CHECKPOINT = "best_model.pth"
EMB_DIM = 8
HORIZON_BIN = 7
W_DIR = 1.0
W_BIN = 0.5
USE_COMPILE = False
FEATURE_NAMES = ["log_ret", "log_mm20", "log_mm50", "bb_haute", "bb_basse",
                 "rsi14", "macd_line", "macd_signal"]
N_BINS_PER_FEATURE = [10, 3, 3, 3, 3, 3, 2, 2]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

all_bin_files = sorted(
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith("_bin.csv")
)
if not all_bin_files:
    print(f"[ERREUR] Aucun fichier _bin.csv trouvé dans {DATA_DIR}/")
    print("Lance d'abord : python tools/Binning.py")
    sys.exit(1)
print(f"{len(all_bin_files)} fichier(s) _bin.csv trouvé(s) dans data/\n")
print(f"Bins par feature : {dict(zip(FEATURE_NAMES, N_BINS_PER_FEATURE))}\n")
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg.endswith("_bin.csv"):
        target_name = arg
    elif arg.endswith(".csv"):
        target_name = arg.replace(".csv", "_bin.csv")
    else:                                   # ex : "AAPL" → "AAPL_bin.csv"
        target_name = arg + "_bin.csv"
    TARGET_FILE = os.path.join(DATA_DIR, target_name)
    if not os.path.exists(TARGET_FILE):
        print(f"[ERREUR] Fichier test introuvable : {TARGET_FILE}")
        sys.exit(1)
else:
    apple_path = os.path.join(DATA_DIR, "apple_bin.csv")
    AAPL_path = os.path.join(DATA_DIR, "AAPL_bin.csv")
    if os.path.exists(apple_path):
        TARGET_FILE = apple_path
    elif os.path.exists(AAPL_path):
        TARGET_FILE = AAPL_path
    else:
        TARGET_FILE = all_bin_files[0]

print(f"Fichier test  : {os.path.basename(TARGET_FILE)}\n")

def edges_path_for(bin_path: str) -> str:
    fname = os.path.basename(bin_path)
    return os.path.join(DATA_DIR, fname.replace("_bin.csv", "_edges.csv"))

def bin7_edges_path_for(bin_path: str) -> str:
    fname = os.path.basename(bin_path)
    return os.path.join(DATA_DIR, fname.replace("_bin.csv", "_bin7_edges.csv"))

def raw_csv_path_for(bin_path: str) -> str:
    fname = os.path.basename(bin_path)
    return os.path.join(DATA_DIR, fname.replace("_bin.csv", ".csv"))

def load_raw_logret(raw_csv: str) -> np.ndarray:
    from data_prep.prepare import prepare_features_with_dates
    features, _ = prepare_features_with_dates(raw_csv)
    return features[:, 0]

all_train_x, all_train_y_dir, all_train_y_bin = [], [], []
all_val_x, all_val_y_dir, all_val_y_bin = [], [], []

for bin_path in all_bin_files:
    fname = os.path.basename(bin_path)
    try:
        ep = edges_path_for(bin_path)
        if not os.path.exists(ep):
            print(f"  [SKIP] {fname} — edges introuvables : {os.path.basename(ep)}")
            continue
        logret_edges = load_logret_edges(ep)

        b7p = bin7_edges_path_for(bin_path)
        bin7_edges = load_bin7_edges(b7p) if os.path.exists(b7p) else None
        if bin7_edges is None:
            print(f"  [WARN] {fname} — bin7_edges introuvables, fallback edges journaliers")

        df = pd.read_csv(bin_path)
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            print(f"  [SKIP] {fname} — colonnes manquantes : {missing}")
            continue
        binned = df[FEATURE_NAMES].values.astype(np.int64)
        n = len(binned)
        if n < WINDOW + HORIZON_BIN + 10:
            print(f"  [SKIP] {fname} — trop peu de lignes ({n})")
            continue
        raw_csv = raw_csv_path_for(bin_path)
        if not os.path.exists(raw_csv):
            print(f"  [SKIP] {fname} — CSV brut introuvable : {os.path.basename(raw_csv)}")
            continue
        raw_logret = load_raw_logret(raw_csv)
        if len(raw_logret) != n:
            print(f"  [SKIP] {fname} — désalignement bin({n}) vs raw({len(raw_logret)})")
            continue
        t_end = int(0.70 * n)
        v_end = int(0.85 * n)
        x_tr, y_dir_tr, y_bin_tr = make_windows_from_binned(
            binned[:t_end], raw_logret[:t_end], logret_edges, WINDOW, HORIZON_BIN, bin7_edges
        )
        x_va, y_dir_va, y_bin_va = make_windows_from_binned(
            binned[t_end:v_end], raw_logret[t_end:v_end], logret_edges, WINDOW, HORIZON_BIN, bin7_edges
        )
        if len(x_tr) == 0 or len(x_va) == 0:
            print(f"  [SKIP] {fname} — fenêtres vides après découpage")
            continue

        all_train_x.append(x_tr); all_train_y_dir.append(y_dir_tr); all_train_y_bin.append(y_bin_tr)
        all_val_x.append(x_va); all_val_y_dir.append(y_dir_va); all_val_y_bin.append(y_bin_va)

        print(f"  ✓ {fname:30s} — train: {x_tr.shape[0]:5d} | val: {x_va.shape[0]:4d} | "
              f"hausse: {y_dir_tr.mean()*100:.1f}% | bin moy: {y_bin_tr.mean():.1f}")

    except Exception as e:
        print(f"  [ERREUR] {fname} : {e}")

if not all_train_x:
    print("\n[ERREUR] Aucun fichier chargé correctement.")
    sys.exit(1)

x_train = np.concatenate(all_train_x); y_dir_train = np.concatenate(all_train_y_dir)
y_bin_train = np.concatenate(all_train_y_bin); x_val = np.concatenate(all_val_x)
y_dir_val = np.concatenate(all_val_y_dir); y_bin_val = np.concatenate(all_val_y_bin)

print(f"\nDataset total — train: {x_train.shape} | val: {x_val.shape}")
print(f"Hausse train: {y_dir_train.mean()*100:.1f}% | Hausse val: {y_dir_val.mean()*100:.1f}%")
print(f"Bin moyen train: {y_bin_train.mean():.2f} | Bin moyen val: {y_bin_val.mean():.2f}")

logret_edges_test = load_logret_edges(edges_path_for(TARGET_FILE))
bin7_edges_test = load_bin7_edges(bin7_edges_path_for(TARGET_FILE)) if os.path.exists(bin7_edges_path_for(TARGET_FILE)) else None
df_test = pd.read_csv(TARGET_FILE)
binned_t = df_test[FEATURE_NAMES].values.astype(np.int64)
n_t = len(binned_t)
raw_logret_t = load_raw_logret(raw_csv_path_for(TARGET_FILE))

test_binned = binned_t[int(0.85 * n_t):]
test_raw_logret = raw_logret_t[int(0.85 * n_t):]

x_test, y_dir_test, y_bin_test = make_windows_from_binned(
    test_binned, test_raw_logret, logret_edges_test, WINDOW, HORIZON_BIN, bin7_edges_test
)
if len(x_test) == 0:
    print(f"[ERREUR] Aucune fenêtre de test pour {os.path.basename(TARGET_FILE)} "
          f"(seulement {len(test_binned)} lignes dans les derniers 15%).")
    sys.exit(1)
print(f"Test ({os.path.basename(TARGET_FILE)}) : {x_test.shape[0]} fenêtres | "
      f"hausse: {y_dir_test.mean()*100:.1f}% | bin moyen: {y_bin_test.mean():.2f}\n")

loader_train, loader_val, loader_test = build_loaders(
    x_train, y_dir_train, y_bin_train,
    x_val, y_dir_val, y_bin_val,
    x_test, y_dir_test, y_bin_test,
    batch_size=BATCH_SIZE,
)
print(f"Nb batches train: {len(loader_train)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"Device : {device} ({torch.cuda.get_device_name(0)})")
    print("Optimisations GPU : cudnn.benchmark + TF32 + AMP fp16\n")
else:
    print(f"Device : {device}\n")
model = build_model(
    n_bins_per_feature=N_BINS_PER_FEATURE,
    emb_dim=EMB_DIM,
    dropout=0.4,
).to(device)
if USE_COMPILE:
    try:
        model = torch.compile(model)
        print("torch.compile activé (CNN1D)")
    except Exception as e:
        print(f"[WARN] torch.compile indisponible : {e}")

history = fit(
    model=model, loader_train=loader_train, loader_val=loader_val,
    device=device, nb_epochs=NB_EPOCHS, lr=LR, patience=PATIENCE,
    w_dir=W_DIR, w_bin=W_BIN,
    checkpoint_path=CHECKPOINT, log_every=10,
)

def evaluate(model, x_test, y_dir_test, y_bin_test, device, target_name,
             eval_batch_size=1024):
    model.eval()
    x_tensor = torch.from_numpy(x_test)
    use_amp = device.type == "cuda"
    logits_dir_parts, logits_bin_parts = [], []
    with torch.inference_mode():
        for chunk in torch.split(x_tensor, eval_batch_size):
            chunk = chunk.to(device, non_blocking=True)
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                ld, lb = model(chunk)
            logits_dir_parts.append(ld.float().cpu())
            logits_bin_parts.append(lb.float().cpu())
    logits_dir = torch.cat(logits_dir_parts).numpy()
    logits_bin = torch.cat(logits_bin_parts).numpy()

    y_pred_dir = np.argmax(logits_dir, axis=1)
    y_pred_bin = np.argmax(logits_bin, axis=1)

    acc_dir = (y_pred_dir == y_dir_test).mean() * 100
    acc_bin = (y_pred_bin == y_bin_test).mean() * 100

    tp = ((y_pred_dir == 1) & (y_dir_test == 1)).sum()
    tn = ((y_pred_dir == 0) & (y_dir_test == 0)).sum()
    fp = ((y_pred_dir == 1) & (y_dir_test == 0)).sum()
    fn = ((y_pred_dir == 0) & (y_dir_test == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n{'='*55}")
    print(f"  Test sur : {target_name}")
    print(f"{'='*55}")

    print(f"\n── Tête 1 : Direction moyenne j+1..j+{HORIZON_BIN} ────────────────────")
    print(f"Accuracy    : {acc_dir:.1f}%  ({(y_pred_dir==y_dir_test).sum()}/{len(y_dir_test)})")
    print(f"Précision   : {prec*100:.1f}%")
    print(f"Recall      : {recall*100:.1f}%")
    print(f"{'':15}  {'Prédit baisse':>13}  {'Prédit hausse':>13}")
    print(f"{'Réel baisse':15}  {tn:>13}  {fp:>13}")
    print(f"{'Réel hausse':15}  {fn:>13}  {tp:>13}")

    print(f"\n── Tête 2 : Bin log_ret moyen j+1..j+{HORIZON_BIN} ──────────────────")
    print(f"Accuracy exacte  : {acc_bin:.1f}%")
    tol1 = (np.abs(y_pred_bin - y_bin_test) <= 1).mean() * 100
    print(f"Accuracy ±1 bin  : {tol1:.1f}%")
    print(f"Erreur moyenne   : {np.abs(y_pred_bin - y_bin_test).mean():.2f} bins")

    labels_dir = {0: "baisse", 1: "hausse"}
    print(f"\n{'#':>4}  {'Dir prédit':>10}  {'Dir réel':>10}  {'Bin préd':>8}  {'Bin réel':>8}  {'OK':>4}")
    print("-" * 55)
    for i in range(min(10, len(y_dir_test))):
        ok = "✓" if y_pred_dir[i] == y_dir_test[i] else "✗"
        print(f"{i+1:>4}  {labels_dir[y_pred_dir[i]]:>10}  {labels_dir[y_dir_test[i]]:>10}  "
              f"{y_pred_bin[i]:>8}  {y_bin_test[i]:>8}  {ok:>4}")

evaluate(model, x_test, y_dir_test, y_bin_test, device, os.path.basename(TARGET_FILE))

from training.bert_mlm import build_bert_model, fit_mlm, fit_finetune

BERT_EMB_DIM = 8
BERT_N_HEADS = 4
BERT_N_LAYERS = 3
BERT_FF_DIM = 256
BERT_DROPOUT = 0.1

MLM_EPOCHS = 50
MLM_LR = 1e-3
MLM_PATIENCE = 10

FT_EPOCHS = 150
FT_LR = 2e-4
FT_PATIENCE = 20

BERT_MLM_CKPT = "bert_pretrain.pth"
BERT_FT_CKPT = "bert_finetune.pth"

print("\n" + "═" * 60)
print("  Instanciation du modèle BERT")
print("═" * 60)
bert_model = build_bert_model(
    n_bins_per_feature=N_BINS_PER_FEATURE,
    emb_dim=BERT_EMB_DIM,
    n_heads=BERT_N_HEADS,
    n_layers=BERT_N_LAYERS,
    ff_dim=BERT_FF_DIM,
    dropout=BERT_DROPOUT,
).to(device)
if USE_COMPILE:
    try:
        bert_model = torch.compile(bert_model)
        print("torch.compile activé (BERT)")
    except Exception as e:
        print(f"[WARN] torch.compile indisponible : {e}")

history_mlm = fit_mlm(
    model=bert_model,
    loader_train=loader_train,
    loader_val=loader_val,
    n_bins_per_feature=N_BINS_PER_FEATURE,
    device=device,
    nb_epochs=MLM_EPOCHS,
    lr=MLM_LR,
    patience=MLM_PATIENCE,
    log_every=5,
    checkpoint_path=BERT_MLM_CKPT,
)

history_ft = fit_finetune(
    model=bert_model,
    loader_train=loader_train,
    loader_val=loader_val,
    device=device,
    nb_epochs=FT_EPOCHS,
    lr=FT_LR,
    patience=FT_PATIENCE,
    w_dir=W_DIR,
    w_bin=W_BIN,
    checkpoint_path=BERT_FT_CKPT,
    log_every=10,
)

print("\n" + "═" * 60)
print("  Évaluation finale — BERT MLM fine-tuné")
print("═" * 60)
evaluate(bert_model, x_test, y_dir_test, y_bin_test, device,
         "BERT_MLM — " + os.path.basename(TARGET_FILE))
