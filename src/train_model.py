import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration
# --- MODIFIED SECTION 1 ---
# We only need the one merged file now
MERGED_DATA_PATH = 'merged_rice_data.csv' 
# --- END MODIFIED SECTION ---

TRAIT = 'GRWT100'  # type of Phenotype (This is from your readme: 'Grain weight')
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
SAVE_PATH = 'best_model.pth'
RANDOM_SEED = 42

# 1. Load the single merged dataset
df = pd.read_csv(MERGED_DATA_PATH)

# 2. Drop missing targets
df = df.dropna(subset=[TRAIT])

# 3. Separate features and target
# --- MODIFIED SECTION 3 ---
# We drop 'Sample_ID' (which we renamed from 'Unnamed: 0')
# and all other phenotype traits.

# First, get the list of *all* phenotype traits from your readme
all_phenotypes = [
    'CUDI_REPRO', 'CULT_REPRO', 'CUNO_REPRO', 'GRLT', 'GRWD',
    'GRWT100', 'HDG_80HEAD', 'LIGLT', 'LLT', 'LWD', 'PLT_POST', 'SDHT'
]

# X = All columns that are NOT 'Sample_ID' and NOT in the phenotype list
# Note: This is a more robust way to select only genotype columns
phenotype_cols_to_drop = list(set(all_phenotypes)) # Get all trait names
cols_to_drop = ['Sample_ID'] + phenotype_cols_to_drop
X = df.drop(columns=cols_to_drop).values  # genotype

# y = The one phenotype trait we want to predict
y = df[TRAIT].values.reshape(-1, 1)          # phenotype
# --- END MODIFIED SECTION ---


# 4. Standardize
scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# 5. Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_scaled, y_scaled, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# 6. Split train+val into train and val
val_size = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_size, random_state=RANDOM_SEED
)

# --- (Rest of the script is identical) ---

class RiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
train_loader = DataLoader(RiceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RiceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(RiceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, loss, optimizer
model = FeedForwardNN(X_train.shape[1], y_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train loop
train_history = []
val_history = []
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_losses.append(loss.item())
    
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    print(f"Epoch {epoch:03d}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
    
    train_history.append(avg_train)
    val_history.append(avg_val)
    
    # Save best
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }, SAVE_PATH)

# Plot training vs. validation loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_history, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_history, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'Training vs. Validation Loss ({TRAIT})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Test evaluation
# Test evaluation
checkpoint = torch.load(SAVE_PATH, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
test_losses = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        test_losses.append(criterion(output, y_batch).item())

print(f"\nFinal Test Loss: {np.mean(test_losses):.4f}")
