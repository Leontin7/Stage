# import pandas as pd
# import numpy as np

# df = pd.read_csv("apple.csv", skiprows=[1, 2])
# df = df.drop_duplicates(subset=["Price"])
# df = df[["Price", "Close"]]
# df = df.sort_values(by="Price")
# df["Close_HA"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
# #np.set_printoptions(threshold=np.inf)
# liste = np.log(df["Close"].iloc[1:].values / df["Close"].iloc[:-1].values)

# print(liste)



# import pandas as pd
# import numpy as np

# df = pd.read_csv("apple.csv", skiprows=[1, 2])
# df = df.rename(columns={"Price": "Date"})
# df = df.drop_duplicates(subset=["Date"])
# df = df.sort_values(by="Date")

# df["Close_HA"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

# log_returns = np.log(df["Close_HA"].iloc[1:].values / df["Close_HA"].iloc[:-1].values)

# print(log_returns)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

df = pd.read_csv("apple.csv", skiprows=[1, 2])
df = df.rename(columns={"Price": "Date"})
df = df.drop_duplicates(subset=["Date"])
df = df.sort_values(by="Date")

df["Close_HA"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

open_ha = np.zeros(len(df))
open_ha[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
for i in range(1, len(df)):
    open_ha[i] = (open_ha[i-1] + df["Close_HA"].iloc[i-1]) / 2
df["Open_HA"] = open_ha

df["High_HA"] = np.maximum(df["High"], np.maximum(df["Open_HA"], df["Close_HA"]))
df["Low_HA"]  = np.minimum(df["Low"],  np.minimum(df["Open_HA"], df["Close_HA"]))

df["SMA20"] = df["Close_HA"].rolling(window=20).mean()
df["SMA50"] = df["Close_HA"].rolling(window=50).mean()
df["EMA20"] = df["Close_HA"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["Close_HA"].ewm(span=50, adjust=False).mean()

df["ecart_SMA20"] = (df["Close_HA"] - df["SMA20"]) / df["Close_HA"]
df["ecart_SMA50"] = (df["Close_HA"] - df["SMA50"]) / df["Close_HA"]
df["ecart_EMA20"] = (df["Close_HA"] - df["EMA20"]) / df["Close_HA"]
df["ecart_EMA50"] = (df["Close_HA"] - df["EMA50"]) / df["Close_HA"]

df = df.dropna()
df = df.reset_index(drop=True)

log_returns = np.log(df["Close_HA"].iloc[1:].values / df["Close_HA"].iloc[:-1].values)

n = len(log_returns)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train = log_returns[:train_end]
val = log_returns[train_end:val_end]
test = log_returns[val_end:]

mu = np.mean(train)
sigma = np.std(train)
train_norm = (train - mu) / sigma
val_norm = (val - mu) / sigma
test_norm = (test - mu) / sigma

def windows(serie, window):
    x = []
    y = []
    for i in range(len(serie) - window):
        x.append(serie[i : i + window])
        y.append(serie[i + window])
    return np.array(x), np.array(y)

window = 20
x_train, y_train = windows(train_norm, window)
x_val, y_val = windows(val_norm, window)
x_test, y_test = windows(test_norm, window)

print(f"x_train: {x_train.shape}")
print(f"x_val: {x_val.shape}")
print(f"x_test: {x_test.shape}")





x_train_np = x_train.astype(np.float32)
x_val_np = x_val.astype(np.float32)
x_test_np = x_test.astype(np.float32)
y_train_np = y_train.astype(np.float32)
y_val_np = y_val.astype(np.float32)
y_test_np = y_test.astype(np.float32)

x_train_t = torch.tensor(x_train_np)
x_val_t = torch.tensor(x_val_np)
x_test_t = torch.tensor(x_test_np)
y_train_t = torch.tensor(y_train_np)
y_val_t = torch.tensor(y_val_np)
y_test_t = torch.tensor(y_test_np)

x_train_t = x_train_t.unsqueeze(2)
x_val_t = x_val_t.unsqueeze(2)
x_test_t = x_test_t.unsqueeze(2)

dataset_train = TensorDataset(x_train_t, y_train_t)
dataset_val = TensorDataset(x_val_t, y_val_t)
dataset_test = TensorDataset(x_test_t, y_test_t)

BATCH_SIZE = len(dataset_train)

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

print(f"x_train_t: {x_train_t.shape}")
print(f"Nb batches train: {len(loader_train)}")

class CNN1D(nn.Module):

    def __init__(self, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, padding='same')
        self.drop1 = nn.Dropout(p=dropout)
        self.bn1   = nn.BatchNorm1d(num_features=100, momentum=0.005)
        self.relu  = nn.ReLU()
        self.fc    = nn.Linear(100, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

model = CNN1D(dropout=0.2)
print(model)
nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Nombre de parametres : {nb_params}")

x_fictif = torch.randn(32, 20, 1)
y_pred = model(x_fictif)
print(f"Forme entree: {x_fictif.shape}")
print(f"Forme sortie: {y_pred.shape}")









device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilise : {device}")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
NB_EPOCHS = 50
for epoch in range(NB_EPOCHS):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in loader_train:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
    train_loss /= len(loader_train)
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader_val:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred  = model(x_batch)
            loss    = criterion(y_pred, y_batch)
            val_loss += loss.item()

    val_loss /= len(loader_val)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{NB_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

x_test_t_pred = torch.tensor(x_test.astype(np.float32)).unsqueeze(2).to(device)

model.eval()
with torch.no_grad():
    y_pred_t = model(x_test_t_pred)

y_pred = y_pred_t.cpu().numpy()
y_reel = y_test

print()
print("=== 10 premieres predictions vs valeurs reelles ===")
print(f"{'Jour':>5}  {'Predit':>10}  {'Reel':>10}  {'Erreur':>10}")
print("-" * 42)
for i in range(10):
    erreur = abs(y_pred[i] - y_reel[i])
    print(f"{i+1:>5}  {y_pred[i]:>10.4f}  {y_reel[i]:>10.4f}  {erreur:>10.4f}")

print()
mse = np.mean((y_pred - y_reel) ** 2)
rmse = np.sqrt(mse)
correct = np.sum(np.sign(y_pred) == np.sign(y_reel))
accuracy = correct / len(y_reel) * 100

print(f"MSE               : {mse:.6f}")
print(f"RMSE              : {rmse:.6f}")
print(f"Directional accuracy : {accuracy:.1f}%  ({correct}/{len(y_reel)} jours)")
