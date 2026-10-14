# cSpell:ignore matplotlib seaborn sklearn numpy pyplot figsize imshow cmap fontsize suptitle xticks savefig bbox ravel argmax argsort softmax relu crossentropy adam onednn mnist keras tensorflow conv maxpooling dropout flatten tkinter ellipse lanczos astype ndarray dtype subplots cbar annot fmt xlabel ylabel ylim xlim rcparams tight layout tqdm
# ============================================================
# CODEALPHA - HANDWRITTEN CHARACTER RECOGNITION  (Task 3)
# A CNN (Convolutional Neural Network) that reads handwritten
# digits (0-9) from the MNIST dataset.
# ============================================================

import os
import sys
import time

# Keep the console output quiet and the results repeatable.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"   # Keras runs on top of TensorFlow

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import keras
from keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)  # same random numbers every run


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, ".write_test"), "w") as f:
        f.write("ok")
    os.remove(os.path.join(RESULTS_DIR, ".write_test"))
except OSError:
    raise SystemExit(
        "Cannot write to the 'results' folder. Move this project folder out of "
        "OneDrive (for example to C:\\Projects) and run the script again."
    )

print("======================================")
print("HANDWRITTEN CHARACTER RECOGNITION")
print("======================================")
print("Project folder :", BASE_DIR)
print("Results folder :", RESULTS_DIR)


# ============================================================
# 2. LOAD THE MNIST DATASET
# (downloads automatically the first time, about 11 MB)
# ============================================================

try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
except Exception as error:   # usually a slow or broken internet connection
    raise SystemExit(
        "Could not download the MNIST dataset. Check your internet connection "
        "and run the script again.\nDetails: " + str(error)
    )

# Tip: run "python handwritten_recognition.py --quick" for a fast 1-minute test
# of your setup (smaller data, 1 epoch). Run without --quick for the real training.
QUICK_TEST = "--quick" in sys.argv
if QUICK_TEST:
    print("\n*** QUICK TEST MODE: only 5,000 training images and 1 epoch ***")
    x_train, y_train = x_train[:5000], y_train[:5000]
    BASELINE_EPOCHS, CNN_EPOCHS = 1, 1
else:
    BASELINE_EPOCHS, CNN_EPOCHS = 8, 12

print("\n========== DATASET ==========")
print("Training images:", x_train.shape)   # (60000, 28, 28) = 60000 pictures of 28x28 pixels
print("Test images    :", x_test.shape)
print("Pixel values   : from", x_train.min(), "to", x_train.max(), "(0 = black, 255 = white)")

print("\nImages per digit (training set):")
print(pd.Series(y_train).value_counts().sort_index().to_string())


# ============================================================
# 3. LOOK AT SOME SAMPLES
# ============================================================

fig, axes = plt.subplots(2, 10, figsize=(14, 3.6))
for digit in range(10):
    examples = np.where(y_train == digit)[0][:2]       # first two images of this digit
    for row, index in enumerate(examples):
        axes[row, digit].imshow(x_train[index], cmap="gray")
        axes[row, digit].set_title(str(digit))
        axes[row, digit].axis("off")
plt.suptitle("Sample images from the MNIST dataset")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sample_digits.png"), dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# 4. PREPARE THE DATA
# ============================================================

# Pixels go from 0-255 to 0-1. Neural networks learn faster on small numbers.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# The CNN expects (height, width, channels). Our pictures are grayscale = 1 channel.
x_train = np.expand_dims(x_train, -1)   # (60000, 28, 28) -> (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

print("\nAfter preparing:", x_train.shape, "| values from", x_train.min(), "to", x_train.max())


# ============================================================
# 5. BUILD TWO MODELS
# ============================================================

def build_baseline():
    """Baseline: a plain neural network. It sees the image as one long row of
    784 pixels, so it has no idea which pixels are neighbours."""
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Flatten(),                       # 28x28 picture -> 784 numbers in a row
            layers.Dense(128, activation="relu"),   # 128 neurons that look at all pixels
            layers.Dense(10, activation="softmax"), # 10 probabilities, one per digit
        ],
        name="baseline_network",
    )


def build_cnn():
    """CNN: scans the picture with small windows, so it can find strokes,
    curves and corners wherever they appear."""
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),  # 32 filters look for simple strokes
            layers.MaxPooling2D((2, 2)),                    # shrink the picture, keep the strongest signals
            layers.Conv2D(64, (3, 3), activation="relu"),  # 64 filters combine strokes into shapes
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),                               # 2D feature maps -> one long row
            layers.Dropout(0.3),                            # randomly switch off 30% of neurons while training (prevents memorising)
            layers.Dense(10, activation="softmax"),         # probability for each digit 0-9
        ],
        name="digit_cnn",
    )


# ============================================================
# 6. TRAIN AND EVALUATE
# ============================================================

def train_and_evaluate(model, epochs):
    """Train the model, then test it on images it has never seen."""
    model.compile(
        optimizer="adam",                         # the method that adjusts the weights
        loss="sparse_categorical_crossentropy",   # the "mistake score" for labels 0-9
        metrics=["accuracy"],
    )

    # Stop early if the validation loss stops improving (prevents over-training).
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    print(f"\n--- Training: {model.name} ---")
    model.summary()

    start = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,            # one epoch = the model sees all training images once
        batch_size=128,           # it learns from 128 images at a time
        validation_split=0.1,     # 10% of training data is kept aside to check progress
        callbacks=[early_stop],
        verbose=2,
    )
    seconds = time.time() - start

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)   # 10 probabilities for every test image
    predictions = probabilities.argmax(axis=1)          # pick the digit with the highest probability

    print(f"{model.name}: test accuracy = {test_accuracy:.4f}, test loss = {test_loss:.4f}")
    return history, seconds, test_loss, test_accuracy, predictions


baseline = build_baseline()
cnn = build_cnn()

base_hist, base_time, base_loss, base_acc, base_pred = train_and_evaluate(baseline, epochs=BASELINE_EPOCHS)
cnn_hist, cnn_time, cnn_loss, cnn_acc, cnn_pred = train_and_evaluate(cnn, epochs=CNN_EPOCHS)


# ============================================================
# 7. COMPARE THE MODELS
# ============================================================

comparison = pd.DataFrame(
    {
        "Parameters": [baseline.count_params(), cnn.count_params()],
        "Training time (s)": [round(base_time, 1), round(cnn_time, 1)],
        "Epochs trained": [len(base_hist.history["loss"]), len(cnn_hist.history["loss"])],
        "Test loss": [round(base_loss, 4), round(cnn_loss, 4)],
        "Test accuracy": [round(base_acc, 4), round(cnn_acc, 4)],
    },
    index=["Baseline network", "CNN"],
)

print("\n========== MODEL COMPARISON ==========")
print(comparison)
comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))


# ============================================================
# 8. TRAINING CURVES
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
epoch_numbers = range(1, len(cnn_hist.history["loss"]) + 1)

ax1.plot(epoch_numbers, cnn_hist.history["accuracy"], marker="o", label="Training")
ax1.plot(epoch_numbers, cnn_hist.history["val_accuracy"], marker="o", label="Validation")
ax1.set_title("CNN accuracy per epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_xticks(list(epoch_numbers))
ax1.legend()

ax2.plot(epoch_numbers, cnn_hist.history["loss"], marker="o", label="Training")
ax2.plot(epoch_numbers, cnn_hist.history["val_loss"], marker="o", label="Validation")
ax2.set_title("CNN loss per epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (lower is better)")
ax2.set_xticks(list(epoch_numbers))
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# 9. CONFUSION MATRIX (where does the CNN mix up digits?)
# ============================================================

cm = confusion_matrix(y_test, cnn_pred)

plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted digit")
plt.ylabel("Actual digit")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# 10. PER-DIGIT REPORT (precision, recall, F1)
# ============================================================

report = classification_report(y_test, cnn_pred, output_dict=True)
report_df = pd.DataFrame(report).T.round(4)
report_df.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))

print("\n========== PER-DIGIT REPORT (CNN) ==========")
print(classification_report(y_test, cnn_pred, digits=4))


# ============================================================
# 11. IMAGES THE CNN GOT WRONG
# ============================================================

wrong = np.where(cnn_pred != y_test)[0]
print("Wrongly classified test images:", len(wrong), "out of", len(y_test))

show = wrong[:25]
if len(show) > 0:
    fig, axes = plt.subplots(5, 5, figsize=(9, 9.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, index in zip(axes.ravel(), show):
        ax.imshow(x_test[index].squeeze(), cmap="gray")
        ax.set_title(f"True {y_test[index]} | Pred {cnn_pred[index]}", fontsize=9)
    plt.suptitle("Mistakes made by the CNN")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "misclassified.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 12. SAVE THE TRAINED CNN
# ============================================================

model_path = os.path.join(MODEL_DIR, "digit_cnn.keras")
cnn.save(model_path)
print("\nModel saved:", model_path)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("FINAL PROJECT SUMMARY")
print("======================================")
print(f"Baseline network test accuracy: {base_acc:.4f}")
print(f"CNN test accuracy             : {cnn_acc:.4f}")
print(f"CNN mistakes                  : {len(wrong)} out of {len(y_test)}")
print("\nPROJECT EXECUTION COMPLETED!")