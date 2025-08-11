import joblib
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Build a Keras model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Fake data
X = np.random.rand(100, 2)
y = X[:, 0] * 3 + X[:, 1] * -2 + 1

# Create pipeline: StandardScaler -> Keras model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KerasRegressor(build_fn=build_model, epochs=5, verbose=0))
])

pipeline.fit(X, y)

# Save as one artifact
joblib.dump(pipeline, "model.joblib")


gcloud ai models upload \
  --region=$REGION \
  --display-name=my-pipeline-model \
  --artifact-uri=gs://$BUCKET_NAME/artifacts \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest
