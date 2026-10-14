# cSpell:ignore matplotlib seaborn sklearn numpy pyplot figsize imshow cmap fontsize suptitle xticks savefig bbox ravel argmax argsort softmax relu crossentropy adam onednn mnist keras tensorflow conv maxpooling dropout flatten tkinter ellipse lanczos astype ndarray dtype subplots cbar annot fmt xlabel ylabel ylim xlim rcparams tight layout tqdm
# ============================================================
# Predict the digit in a picture file.
# Usage:  python predict_digit.py my_digit.png
# ============================================================

import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from PIL import Image
import keras

from digit_utils import prepare_digit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "digit_cnn.keras")

if len(sys.argv) < 2:
    raise SystemExit("Usage: python predict_digit.py <image file>")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run handwritten_recognition.py first.")

model = keras.models.load_model(MODEL_PATH)

image = Image.open(sys.argv[1])
digit = prepare_digit(image, photo=True)   # photo=True removes shadows and uneven light
if digit is None:
    raise SystemExit("Could not find a digit in this picture.")

# Save what the model actually sees (a 28x28 picture), handy for demos and debugging.
results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)
Image.fromarray((digit * 255).astype("uint8")).resize((224, 224), Image.Resampling.NEAREST).save(
    os.path.join(results_dir, "last_prediction_input.png")
)

probabilities = model.predict(digit.reshape(1, 28, 28, 1), verbose=0)[0]
best = np.argsort(probabilities)[::-1][:3]

print(f"\nPredicted digit: {best[0]}  (confidence {probabilities[best[0]] * 100:.1f}%)")
print("Top 3 guesses:")
for d in best:
    print(f"  {d}: {probabilities[d] * 100:5.1f}%")