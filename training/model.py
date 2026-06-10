"""
training/model.py
=================
Architecture CNN1D + BiLSTM + LSTM — double tête de prédiction :
  - Tête 1 : hausse/baisse du log_ret moyen j+1..j+7 → logits (batch, 2)
  - Tête 2 : bin du log_ret moyen sur j+1..j+7       → logits (batch, n_bins_logret)

Entrée : fenêtre (batch, window, F) de bin_ids en LONG
         Chaque feature f a son propre nn.Embedding(n_bins_f, emb_dim).
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(
        self,
        n_bins_per_feature: list[int],
        emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        """
        Paramètres
        ----------
        n_bins_per_feature : liste du nombre de bins pour chaque feature,
                             dans le même ordre que FEATURE_NAMES
        emb_dim            : dimension de l'embedding pour chaque bin
        dropout            : taux de dropout
        """
        super().__init__()

        self.n_features = len(n_bins_per_feature)
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_bins, emb_dim)
            for n_bins in n_bins_per_feature
        ])
        conv_in = self.n_features * emb_dim

        self.conv1 = nn.Conv1d(conv_in, 100, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(100, momentum=0.1)
        self.drop1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(100, 100, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(100, momentum=0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv1d(100, 100, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(100, momentum=0.1)
        self.drop3 = nn.Dropout(p=dropout)

        self.conv4 = nn.Conv1d(100, 100, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm1d(100, momentum=0.1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop4 = nn.Dropout(p=dropout)

        self.bilstm = nn.LSTM(100, 200, batch_first=True, bidirectional=True)
        self.drop5 = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(400, 200, batch_first=True, bidirectional=False)
        self.drop6 = nn.Dropout(p=dropout)

        self.fc_shared = nn.Linear(200, 100)
        self.head_direction = nn.Linear(100, 2)
        n_bins_logret = n_bins_per_feature[0]
        self.head_bin = nn.Linear(100, n_bins_logret)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        """
        x : (batch, window, n_features) — dtype=torch.long
        Retourne (logits_dir, logits_bin)
        """
        emb_list = [
            self.embeddings[f](x[:, :, f])
            for f in range(self.n_features)
        ]
        x = torch.cat(emb_list, dim=-1)
        x = x.permute(0, 2, 1)
        x = self.drop1(self.act(self.bn1(self.conv1(x))))
        x = self.drop2(self.pool1(self.act(self.bn2(self.conv2(x)))))
        x = self.drop3(self.act(self.bn3(self.conv3(x))))
        x = self.drop4(self.pool2(self.act(self.bn4(self.conv4(x)))))

        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.drop5(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.drop6(x)

        shared = self.act(self.fc_shared(x))
        logits_dir = self.head_direction(shared)
        logits_bin = self.head_bin(shared)

        return logits_dir, logits_bin


def build_model(
    n_bins_per_feature: list[int],
    emb_dim: int = 8,
    dropout: float = 0.2,
) -> CNN1D:
    model = CNN1D(n_bins_per_feature=n_bins_per_feature,
                  emb_dim=emb_dim, dropout=dropout)
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle CNN1D instancié — {nb_params:,} paramètres entraînables")
    print(f"  bins/feature : {n_bins_per_feature}")
    print(f"  emb_dim={emb_dim} | conv_in={len(n_bins_per_feature)*emb_dim} | dropout={dropout}")
    print(f"  Sorties : head_direction (2) + head_bin ({n_bins_per_feature[0]})")
    return model
