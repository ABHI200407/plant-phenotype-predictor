import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- STEP 1: REPLICATE DATA SETUP ---
# We need to load the data again to know:
# 1. The input shape (X.shape[1])
# 2. The scaling parameters (StandardScaler)
print("Loading data to determine model shape...")

df = pd.read_csv('merged_rice_data.csv')
df.fillna(0, inplace=True)

# Define target and drops exactly like in your training script
y = df['GRWT100'].values
drop_cols = ['Sample_ID', 'GRWT100', 'GRLT', 'GRWD', 'HDG_80HEAD', 
             'LIGLT', 'LLT', 'LWD', 'PLT_POST', 'SDHT', 'CUNO_REPRO']
existing_drops = [c for c in drop_cols if c in df.columns]
X = df.drop(existing_drops, axis=1).values

# We need to fit the scaler so our test data matches the training scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train) 

# Get the number of input columns
input_features = X.shape[1]
print(f"Detected {input_features} input features.")

# --- STEP 2: DEFINE ARCHITECTURE ---
# This must match your training file exactly
model = nn.Sequential(
    nn.Linear(input_features, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# --- STEP 3: LOAD WEIGHTS ---
print("Loading best_model.pth...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
print("Model loaded successfully!")

# --- STEP 4: TEST PREDICTION ---
print("\n--- TESTING ON A REAL SAMPLE ---")

# Let's take the first row from your dataset to test
sample_data_raw = X[0] 
actual_value = y[0]

# We must scale the input just like we did in training
sample_data_scaled = scaler.transform([sample_data_raw])
sample_tensor = torch.FloatTensor(sample_data_scaled)

# Run prediction
with torch.no_grad():
    prediction = model(sample_tensor)

print(f"Actual GRWT100 Value:    {actual_value}")
print(f"Predicted GRWT100 Value: {prediction.item():.4f}")