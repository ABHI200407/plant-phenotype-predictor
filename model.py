import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load & Clean
df = pd.read_csv('merged_rice_data.csv')
df.fillna(0, inplace=True)

y = df['GRWT100'].values
drop_cols = ['Sample_ID', 'GRWT100', 'GRLT', 'GRWD', 'HDG_80HEAD', 
             'LIGLT', 'LLT', 'LWD', 'PLT_POST', 'SDHT', 'CUNO_REPRO']
existing_drops = [c for c in drop_cols if c in df.columns]
X = df.drop(existing_drops, axis=1).values

# 2. Preprocessing for Neural Network
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.FloatTensor(y_train).view(-1, 1)

# 3. Define Model Architecture
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 4. Training Loop (Epochs)
print("Starting Training with Epochs...")
epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    # Print progress every 5 epochs
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}')

# 5. Save
torch.save(model.state_dict(), 'best_model.pth')
print("SUCCESS: Model saved as 'best_model.pth'")