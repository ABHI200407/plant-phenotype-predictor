import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# --- 1. DATA COLLECTION & PREPROCESSING ---
print("Loading data...")
# Make sure you have the CSV file in the same folder, or provide the full path
# We select only the columns we need for this simple model
try:
    data = pd.read_csv('crop_yield.csv', usecols=[
        'State', 'Season', 'Crop', 'Annual_Rainfall', 'Fertilizer', 'Yield'
    ])
except FileNotFoundError:
    print("Error: 'crop_yield.csv' not found. Make sure it's in the same directory.")
    exit()

print("Cleaning data...")
# Drop any rows where we have missing data for these columns
data = data.dropna()

# --- 2. FEATURE SELECTION (Defining X and y) ---
# These are the "features" (inputs) our model will learn from
# We separate the columns that are text (categorical) from the numbers (numerical)
categorical_features = ['State', 'Season', 'Crop']
numerical_features = ['Annual_Rainfall', 'Fertilizer']

# This is the "target" (output) we want to predict
target = 'Yield'

X_cat = data[categorical_features]
X_num = data[numerical_features]
y = data[target]


# --- 3. PREPROCESSING (Encoding) ---
# Models can't read text ("Punjab"), so we convert text to numbers
# We use OneHotEncoder, which turns "State" into "State_Punjab", "State_Kerala", etc.
print("Encoding features...")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)

# Combine our new encoded text columns with our original number columns
# This gives us our final feature set, X
X = pd.concat([
    pd.DataFrame(X_cat_encoded, columns=encoder.get_feature_names_out()),
    X_num.reset_index(drop=True)
], axis=1)

print(f"Data ready. Total features: {X.shape[1]}")


# --- 4. TRAIN/TEST SPLIT ---
# Split the data: 80% to train the model, 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 5. MODEL TRAINING ---
print("Training Random Forest model...")
# We are using RandomForestRegressor because "Yield" is a number (regression)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


# --- 6. MODEL EVALUATION (Testing) ---
print("Evaluating model...")
predictions = model.predict(X_test)

# R-squared is a good metric. 1.0 is a perfect score. 0.0 is a terrible score.
# This score will tell you how much of the yield variation is explained by your features.
score = r2_score(y_test, predictions)
print(f"Model Evaluation complete. R-squared score: {score:.4f}")


# --- 7. SAVE THE MODEL (.pkl) ---
# This is the final step you wanted!
# We save the trained model AND the encoder to a file.
# We need to save the 'encoder' so the API can preprocess new user data the *exact same way*.
joblib.dump(model, 'yield_model.pkl')
joblib.dump(encoder, 'yield_encoder.pkl')

print("\n--- SUCCESS! ---")
print("Model saved as 'yield_model.pkl'")
print("Encoder saved as 'yield_encoder.pkl'")
