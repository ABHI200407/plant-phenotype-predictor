import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. LOAD DATA
# ==========================================
filename = 'merged_rice_data.csv'
print(f"Loading '{filename}'...")

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"ERROR: Could not find {filename}. Make sure it is in the same folder.")
    exit()

# Fill any empty genetic data with 0
df.fillna(0, inplace=True)

# ==========================================
# 2. CLEANING & PREPARATION
# ==========================================

# A. Define the Target (The Answer)
# We are predicting Grain Weight
if 'GRWT100' not in df.columns:
    print("ERROR: Target column 'GRWT100' not found in CSV.")
    exit()

y = df['GRWT100']

# B. Define Features (The DNA)
# We remove the ID (String) and other physical traits (which would be 'cheating')
columns_to_remove = [
    'Sample_ID',   # The string ID
    'GRWT100',     # The answer (Target)
    # Other physical traits to exclude so we rely ONLY on DNA:
    'GRLT', 'GRWD', 'HDG_80HEAD', 'LIGLT', 'LLT', 
    'LWD', 'PLT_POST', 'SDHT', 'CUNO_REPRO'
]

# Safely drop columns only if they exist
existing_drops = [col for col in columns_to_remove if col in df.columns]
X = df.drop(existing_drops, axis=1)

# C. Feature Encoding Check
# If your genetic data is still Strings (A, T, G, C), we must convert them to numbers.
# If your data is already 0, 1, 2, this block will simply skip.
print("Checking for text data...")
X = pd.get_dummies(X, drop_first=True)

print(f"Training Data Ready.")
print(f"Features (DNA Markers): {X.shape[1]}")
print(f"Target: GRWT100 (Grain Weight)")

# ==========================================
# 3. TRAIN (UPGRADED ENGINE)
# ==========================================
print("\nStarting Training with Gradient Boosting... (This is smarter than Random Forest)")

# Split data: 80% to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# UPGRADE: Using GradientBoostingRegressor
# This builds trees one by one, where each new tree tries to fix the errors of the previous one.
model = GradientBoostingRegressor(
    n_estimators=200,     # Number of boosting stages to perform
    learning_rate=0.1,    # Shrinks the contribution of each tree (prevents overfitting)
    max_depth=3,          # Limits the number of nodes in the tree
    random_state=42,
    verbose=1             # Prints progress
)

model.fit(X_train, y_train)

# ==========================================
# 4. TEST RESULTS
# ==========================================
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"FINAL REPORT CARD")
print("-" * 30)
print(f"Accuracy (R2 Score): {r2:.4f}")
print(f"Average Error (MAE): {mae:.4f}")
print("-" * 30)

if r2 < 0.3:
    print("NOTE: Accuracy is low. This usually means the 10 genetic markers")
    print("selected do not strongly control Grain Weight, or more data is needed.")
else:
    print("Great! The model found a signal in the genetics.")

# ==========================================
# 5. SAVE (Crucial for your Backend)
# ==========================================
joblib.dump(model, 'pheno_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')

print("\nFILES SAVED SUCCESSFULLY:")
print("1. pheno_model.pkl (The Brain)")
print("2. model_features.pkl (The List of Genes)")