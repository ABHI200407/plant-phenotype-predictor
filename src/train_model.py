print("🚀 Training ML model for plant phenotypes...")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Fake data: 100 plants, 5 genomic features (e.g., gene variants)
X = np.random.rand(100, 5) * 10  # "DNA" inputs
y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.rand(100) * 5  # Output: heights ~0-50 cm

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Test
preds = model.predict(X_test)
print(f"Trained! Error: {mean_squared_error(y_test, preds):.2f}")
joblib.dump(model, '../models/phenotype_model.joblib')
print("Saved to models/—ML ready for predictions!")
