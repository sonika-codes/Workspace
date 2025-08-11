import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# === 1. Create fake data ===
X = np.random.rand(100, 2)
y = X[:, 0] * 3 + X[:, 1] * -2 + 1

# === 2. Preprocess ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_scaled, y, epochs=10, verbose=0)

# === 4. Save artifacts ===
os.makedirs("artifacts/keras_model", exist_ok=True)
scaler_path = "artifacts/scaler.pkl"
joblib.dump(scaler, scaler_path)
model.save("artifacts/keras_model")


import joblib
import numpy as np
import tensorflow as tf
from typing import List

class Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def load(self):
        # Load scaler and model
        self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
        self.model = tf.keras.models.load_model(f"{self.model_dir}/keras_model")

    def predict(self, instances: List) -> List:
        # Convert to array
        X = np.array(instances)
        # Preprocess
        X_scaled = self.scaler.transform(X)
        # Predict
        preds = self.model.predict(X_scaled)
        return preds.flatten().tolist()
