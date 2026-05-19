"""
training/model.py
=================
Architecture du modèle CNN1D + BiLSTM + LSTM.

Structure :
    - 4 blocs Conv1D  (Conv → BN → Swish → Dropout)
    - 2 blocs MaxPool après les blocs 2 et 4
    - 1 BiLSTM  (200 unités par direction → sortie 400)
    - 1 LSTM    (200 unités)
    - 2 couches FC  (200 → 100 → 1)

Activation : Swish (nn.SiLU) — meilleure convergence que ReLU,
             gère les gradients pour les entrées <= 0.
BatchNorm  : momentum=0.1, adapté à des datasets de ~10k–20k samples.
Ordre      : Conv → BN → Activation → Dropout (BN avant activation
             pour normaliser les sorties brutes de la conv).
"""

import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    Réseau CNN1D + BiLSTM + LSTM pour la prédiction de séries temporelles.

    Paramètres
    ----------
    n_features : nombre de features en entrée (défaut 8)
    dropout    : taux de dropout appliqué après chaque bloc (défaut 0.2)
    """

    def __init__(self, n_features: int = 8, dropout: float = 0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(n_features, 100, kernel_size=3, padding="same")
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

        self.fc = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 1)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du réseau.

        Paramètres
        ----------
        x : tenseur (batch, window, n_features)

        Retourne
        --------
        tenseur (batch,) — log-return prédit
        """
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

        x = self.act(self.fc(x))
        x = self.fc2(x).squeeze(1)
        return x


def build_model(n_features: int = 8, dropout: float = 0.2) -> CNN1D:
    """
    Instancie et retourne le modèle CNN1D.
    Affiche le nombre de paramètres entraînables.

    Paramètres
    ----------
    n_features : nombre de features en entrée
    dropout    : taux de dropout

    Retourne
    --------
    model : instance de CNN1D
    """
    model = CNN1D(n_features=n_features, dropout=dropout)
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle CNN1D instancié — {nb_params:,} paramètres entraînables")
    return model
