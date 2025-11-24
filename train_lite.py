# File: train_lite.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("1. Loading Data...")
df = pd.read_csv('merged_rice_data.csv')
df.fillna(0, inplace=True)

# --- CONFIGURATION FOR LITE MODEL ---
# These are the 10 columns your website uses.
# Make sure these match your CSV column names exactly.
target_col = 'GRWT100'
lite_features = [
    df.columns[12], df.columns[13], df.columns[14], df.columns[15], df.columns[16],
    df.columns[17], df.columns[18], df.columns[19], df.columns[20], df.columns[21]
]

print(f"Selected 10 Features: {lite_features}")

X = df[lite_features].values
y = df[target_col].values

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Save the Scaler (Crucial for the Lite Model)
joblib.dump(scaler, 'lite_scaler.pkl')

# Convert to Tensor
X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.FloatTensor(y_train).view(-1, 1)

# Define Lite Architecture (Inputs = 10)
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("2. Training Lite Model...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Save the Lite Model
torch.save(model.state_dict(), 'lite_model.pth')
print("SUCCESS: Created 'lite_model.pth' and 'lite_scaler.pkl'")