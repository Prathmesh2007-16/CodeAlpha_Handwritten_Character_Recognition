# cSpell:ignore matplotlib seaborn sklearn numpy pyplot figsize imshow cmap fontsize suptitle xticks savefig bbox ravel argmax argsort softmax relu crossentropy adam onednn mnist keras tensorflow conv maxpooling dropout flatten tkinter ellipse lanczos astype ndarray dtype subplots cbar annot fmt xlabel ylabel ylim xlim rcparams tight layout tqdm
# ============================================================
# Draw a digit with your mouse and let the CNN guess it.
# Usage:  python draw_and_predict.py
# ============================================================

import os
import tkinter as tk

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from PIL import Image, ImageDraw
import keras

from digit_utils import prepare_digit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "digit_cnn.keras")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run handwritten_recognition.py first.")

model = keras.models.load_model(MODEL_PATH)

SIZE = 280        # drawing area in pixels
BRUSH = 11        # brush radius


class DrawingApp:
    def __init__(self, root):
        root.title("Handwritten digit recognition")
        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=12, pady=12)

        # A hidden copy of the drawing that we can feed to the model.
        self.image = Image.new("L", (SIZE, SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)   # drawing while the mouse button is held
        self.canvas.bind("<Button-1>", self.paint)    # a single click also makes a dot

        tk.Button(root, text="Predict", width=14, command=self.predict).grid(row=1, column=0, pady=4)
        tk.Button(root, text="Clear", width=14, command=self.clear).grid(row=1, column=1, pady=4)

        self.result = tk.Label(root, text="Draw a digit (0-9) and press Predict",
                               font=("Arial", 13), justify="left")
        self.result.grid(row=2, column=0, columnspan=2, pady=10)

    def paint(self, event):
        x1, y1, x2, y2 = event.x - BRUSH, event.y - BRUSH, event.x + BRUSH, event.y + BRUSH
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.pen.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.pen.rectangle([0, 0, SIZE, SIZE], fill=0)
        self.result.config(text="Draw a digit (0-9) and press Predict")

    def predict(self):
        digit = prepare_digit(self.image)
        if digit is None:
            self.result.config(text="Please draw a digit first")
            return
        probabilities = model.predict(digit.reshape(1, 28, 28, 1), verbose=0)[0]
        best = np.argsort(probabilities)[::-1][:3]
        lines = [f"Prediction: {best[0]}   ({probabilities[best[0]] * 100:.1f}% sure)", ""]
        lines += [f"{d}:  {probabilities[d] * 100:5.1f}%" for d in best]
        self.result.config(text="\n".join(lines))


root = tk.Tk()
DrawingApp(root)
root.mainloop()