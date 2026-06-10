"""
visualize_bert.py
=================
Visualise les rapprochements appris par le modèle BERT après entraînement.

Ce que ça montre
----------------
1. Similarité entre bins d'une même feature
   → Est-ce que le modèle a appris que bin_5 ressemble à bin_6 ?

2. Projection 2D des embeddings (PCA)
   → Les bins proches dans l'espace appris sont-ils proches dans la réalité ?

3. Corrélations entre features
   → Quelles features le modèle associe-t-il ensemble ?

4. Importance de chaque feature pour la prédiction finale
   → Quelles features pèsent le plus dans head_bin ?

Usage
-----
    python visualize_bert.py                    # utilise bert_finetune.pth
    python visualize_bert.py bert_pretrain.pth  # utilise le pré-entraînement
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.bert_mlm import build_bert_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "bert_finetune.pth"

FEATURE_NAMES      = ["log_ret", "log_mm20", "log_mm50", "bb_haute",
                       "bb_basse", "rsi14", "macd_line", "macd_signal"]
N_BINS_PER_FEATURE = [10, 3, 3, 3, 3, 3, 2, 2]

BERT_EMB_DIM  = 8
BERT_N_HEADS  = 4
BERT_N_LAYERS = 3
BERT_FF_DIM   = 256
BERT_DROPOUT  = 0.1

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

print(f"Chargement du modèle depuis : {CHECKPOINT}")
model = build_bert_model(
    n_bins_per_feature=N_BINS_PER_FEATURE,
    emb_dim=BERT_EMB_DIM,
    n_heads=BERT_N_HEADS,
    n_layers=BERT_N_LAYERS,
    ff_dim=BERT_FF_DIM,
    dropout=BERT_DROPOUT,
)
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
print("Modèle chargé.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Extraction des embeddings
# ─────────────────────────────────────────────────────────────────────────────

embeddings = []
for f, name in enumerate(FEATURE_NAMES):
    emb = model.embeddings[f].weight.detach().numpy()  # (n_bins, emb_dim)
    embeddings.append(emb)
    print(f"  {name:15s} : {emb.shape[0]} bins × {emb.shape[1]} dims")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Similarité cosinus entre bins de log_ret
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(A):
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    A_n = A / norms
    return A_n @ A_n.T

fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
fig.suptitle("Similarité cosinus entre bins (par feature)\n"
             "Plus c'est vert = plus le modèle considère ces bins comme similaires",
             fontsize=13, fontweight="bold")

for f, (name, emb) in enumerate(zip(FEATURE_NAMES, embeddings)):
    ax = axes[f // 4][f % 4]
    sim = cosine_sim(emb)
    n = emb.shape[0]
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"{name}\n({n} bins)", fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")

plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="Similarité cosinus")
# Pas de tight_layout ici : incompatible avec une colorbar partagée
# (UserWarning + chevauchement). constrained_layout gère l'espacement.
out1 = os.path.join(OUT_DIR, "bert_similarite_bins.png")
plt.savefig(out1, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n✓ Similarité bins sauvegardée : {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Projection PCA 2D des embeddings de log_ret
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Projection PCA 2D des embeddings par feature\n"
             "Les bins proches dans ce graphe sont traités de façon similaire par le modèle",
             fontsize=13, fontweight="bold")

colors = plt.cm.RdYlGn(np.linspace(0, 1, 10))

for f, (name, emb) in enumerate(zip(FEATURE_NAMES, embeddings)):
    ax = axes[f // 4][f % 4]
    n = emb.shape[0]

    if n >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb)
        var = pca.explained_variance_ratio_
        # Pour les features à 2 bins, PC2 ne porte aucune variance : ses
        # coordonnées ne sont que du bruit numérique (~1e-7 ± 1e-14) qui
        # rend l'axe illisible (notation offset). On force ces axes à 0.
        for k in range(coords.shape[1]):
            if var[k] < 1e-12:
                coords[:, k] = 0.0
    else:
        coords = np.array([[0, 0]] * n)
        var = [0, 0]

    for i in range(n):
        c = colors[i] if n == 10 else plt.cm.Set1(i / max(n - 1, 1))
        ax.scatter(coords[i, 0], coords[i, 1], s=200, color=c, zorder=3)
        ax.annotate(f"bin {i}", (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Relie les bins consécutifs
    for i in range(n - 1):
        ax.plot([coords[i, 0], coords[i+1, 0]],
                [coords[i, 1], coords[i+1, 1]],
                "gray", alpha=0.4, linewidth=1)

    ax.set_title(f"{name} ({n} bins)\nPCA var: {var[0]*100:.0f}%+{var[1]*100:.0f}%",
                 fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "bert_pca_embeddings.png")
plt.savefig(out2, dpi=120, bbox_inches="tight")
plt.close()
print(f"✓ PCA embeddings sauvegardée  : {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Importance des features pour head_bin
# ─────────────────────────────────────────────────────────────────────────────

# Les embeddings de chaque feature forment un bloc de emb_dim dimensions
# dans le vecteur d_model. On regarde la norme des poids de fc_shared
# pour chaque bloc → importance de chaque feature.

fc_w = model.fc_shared.weight.detach().numpy()  # (128, d_model)
d_model = fc_w.shape[1]
emb_dim = BERT_EMB_DIM
n_feat = len(FEATURE_NAMES)

importance = []
for f in range(n_feat):
    start = f * emb_dim
    end = start + emb_dim
    imp = np.linalg.norm(fc_w[:, start:end])
    importance.append(imp)

importance = np.array(importance)
importance = importance / importance.sum() * 100

fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = plt.cm.RdYlGn(importance / importance.max())
bars = ax.bar(FEATURE_NAMES, importance, color=colors_bar, edgecolor="black", linewidth=0.5)
ax.set_title("Importance de chaque feature pour la prédiction du bin moyen\n"
             "(norme des poids fc_shared par bloc de feature)", fontsize=12)
ax.set_ylabel("Importance relative (%)")
ax.set_ylim(0, importance.max() * 1.2)
for bar, val in zip(bars, importance):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "bert_importance_features.png")
plt.savefig(out3, dpi=120, bbox_inches="tight")
plt.close()
print(f"✓ Importance features sauvegardée : {out3}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Attention moyenne du Transformer (dernière couche)
# ─────────────────────────────────────────────────────────────────────────────
# On passe une séquence type et on regarde vers quelles positions
# le token [CLS] porte son attention.

print("\nCalcul des patterns d'attention...")

# Séquence synthétique : bin médian de chaque feature
# ([10,3,3,3,3,3,2,2] → [5,1,1,1,1,1,1,1])
mid_bins = [nb // 2 for nb in N_BINS_PER_FEATURE]
x = torch.tensor([[mid_bins] * 20], dtype=torch.long)

# Hook pour capturer l'attention
attention_maps = []

def hook_fn(module, input, output):
    # TransformerEncoderLayer — on accède à l'attention via self_attn
    pass

# Utilise forward_pretrain et extrait manuellement via le modèle
with torch.no_grad():
    B, T, F = x.shape
    embs = [model.embeddings[f](x[:, :, f]) for f in range(F)]
    tok_emb = torch.cat(embs, dim=-1)
    cls = model.cls_token.expand(B, -1, -1)
    seq = torch.cat([cls, tok_emb], dim=1)
    positions = torch.arange(1 + T).unsqueeze(0)
    seq = seq + model.pos_emb(positions)

    # Passe dans chaque couche et capture l'attention de la dernière
    hidden = seq
    last_attn = None
    for layer in model.transformer.layers:
        # Le modèle est en Pre-LN (norm_first=True) : dans le vrai forward,
        # l'attention est calculée sur norm1(hidden), pas sur hidden brut.
        # Sans cette normalisation, les poids capturés ne correspondent pas
        # à ce que le modèle calcule réellement.
        src = layer.norm1(hidden) if layer.norm_first else hidden
        attn_out, attn_weights = layer.self_attn(
            src, src, src,
            need_weights=True, average_attn_weights=True
        )
        last_attn = attn_weights.squeeze(0).detach().numpy()  # (T+1, T+1)
        # Continue le forward normalement
        hidden = layer(hidden)

if last_attn is not None:
    # Attention du [CLS] vers les positions temporelles
    cls_attention = last_attn[0, 1:]  # (T,) — [CLS] vers chaque jour

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Patterns d'attention — dernière couche Transformer",
                 fontsize=13, fontweight="bold")

    # Heatmap complète
    ax = axes[0]
    im = ax.imshow(last_attn, cmap="Blues", aspect="auto")
    ax.set_title("Matrice d'attention complète\n(ligne i = position i regarde position j)")
    ax.set_xlabel("Position source (j)")
    ax.set_ylabel("Position cible (i)")
    ax.set_xticks([0] + list(range(1, 21, 4)))
    ax.set_xticklabels(["CLS"] + [f"j{i}" for i in range(1, 21, 4)], fontsize=8)
    ax.set_yticks([0] + list(range(1, 21, 4)))
    ax.set_yticklabels(["CLS"] + [f"j{i}" for i in range(1, 21, 4)], fontsize=8)
    plt.colorbar(im, ax=ax)

    # Attention du [CLS] uniquement
    ax = axes[1]
    days = list(range(1, T + 1))
    ax.bar(days, cls_attention, color=plt.cm.Blues(cls_attention / cls_attention.max()),
           edgecolor="black", linewidth=0.3)
    ax.set_title("[CLS] regarde quels jours ?\n(plus c'est haut = plus le modèle y porte attention)")
    ax.set_xlabel("Jour dans la fenêtre (1=le plus ancien, 20=le plus récent)")
    ax.set_ylabel("Poids d'attention")
    ax.set_xticks(days)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out4 = os.path.join(OUT_DIR, "bert_attention.png")
    plt.savefig(out4, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Attention sauvegardée         : {out4}")

print("\n" + "═" * 55)
print("  Résumé des rapprochements appris")
print("═" * 55)
print(f"\nFeature la plus importante : {FEATURE_NAMES[np.argmax(importance)]} ({importance.max():.1f}%)")
print(f"Feature la moins importante: {FEATURE_NAMES[np.argmin(importance)]} ({importance.min():.1f}%)")

# Rapprochements de bins pour log_ret
# NB : on neutralise la diagonale différemment pour argmax et argmin,
# sinon argmin retombe toujours sur la valeur sentinelle de la diagonale
# et affiche "bin 0 ↔ bin 0 (sim=-1.000)".
emb_logret = embeddings[0]
sim = cosine_sim(emb_logret)

sim_max = sim.copy()
np.fill_diagonal(sim_max, -np.inf)
i, j = np.unravel_index(sim_max.argmax(), sim_max.shape)
print(f"\nBins log_ret les plus similaires : bin {i} ↔ bin {j}  (sim={sim[i, j]:.3f})")

sim_min = sim.copy()
np.fill_diagonal(sim_min, np.inf)
i2, j2 = np.unravel_index(sim_min.argmin(), sim_min.shape)
print(f"Bins log_ret les plus opposés    : bin {i2} ↔ bin {j2}  (sim={sim[i2, j2]:.3f})")

print(f"\n4 images sauvegardées dans : {OUT_DIR}/")
print("  - bert_similarite_bins.png")
print("  - bert_pca_embeddings.png")
print("  - bert_importance_features.png")
print("  - bert_attention.png")
