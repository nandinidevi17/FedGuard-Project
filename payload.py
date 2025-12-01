# server_expected_shapes.py
import os, numpy as np, tensorflow as tf
from config import MODELS_DIR

model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
m = tf.keras.models.load_model(model_path, compile=False)
w = m.get_weights()
print("SERVER: expected arrays:", len(w))
for i, arr in enumerate(w):
    print(f"{i:02d}", np.array(arr).shape)
print("Total params:", sum(np.array(arr).size for arr in w))
