# ✍️ Handwritten Digit Recognition with a CNN

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20accuracy-99.07%25-2ea44f?style=flat-square)
![Task](https://img.shields.io/badge/CodeAlpha-ML%20Internship%20Task%203-7c5cff?style=flat-square)

A **Convolutional Neural Network (CNN)** that reads handwritten digits (0 to 9), built as **Task 3** of the **CodeAlpha Machine Learning internship**. It reaches **99.07% accuracy** on 10,000 test images it has never seen, and it comes with a small app where you draw a digit with your mouse and the model guesses it live.

## 🎯 Objective

Identify handwritten characters from images using image processing and deep learning. This project uses the **MNIST** digit dataset and a CNN, as suggested in the task description.

## 📂 Dataset

- **MNIST:** 60,000 training images and 10,000 test images, each a 28 x 28 pixel grayscale picture of one digit
- The classes are well balanced, from 5,421 images (digit 5) to 6,742 images (digit 1) per digit in the training set
- Downloaded automatically by Keras the first time the script runs (about 11 MB)

<p align="center">
  <img src="results/sample_digits.png" width="80%" alt="Sample MNIST digits"/>
</p>

## 🛠️ Approach

1. **Load and look at the data:** shapes, pixel range and number of images per digit
2. **Prepare the data:** pixel values are scaled from 0-255 to 0-1, and each image gets a channel dimension (28 x 28 x 1)
3. **Baseline model:** a plain neural network (one hidden layer) that sees the image as one long row of 784 pixels
4. **CNN model:** two convolution layers that scan the image for strokes and shapes, pooling layers that shrink it, dropout to avoid memorising, and a final softmax layer that outputs a probability for each digit
5. **Training:** the Adam optimizer, with 10% of the training data kept aside for validation and **early stopping** to stop when the validation loss stops improving
6. **Evaluation:** accuracy, per-digit precision, recall and F1, a confusion matrix, and a look at the images the CNN got wrong

```
Input 28x28x1
  -> Conv2D (32 filters, 3x3) -> MaxPooling (2x2)
  -> Conv2D (64 filters, 3x3) -> MaxPooling (2x2)
  -> Flatten -> Dropout (0.3)
  -> Dense (10, softmax)          = probability of digits 0-9
```

## 📊 Results

The CNN beats the baseline network while using about **one third of the parameters**:

| Model | Parameters | Epochs trained | Test accuracy | Mistakes on 10,000 images |
|---|---|---|---|---|
| Baseline network | 101,770 | 8 | 97.28% | about 272 |
| **CNN** | **34,826** | **9** (early stopping) | **99.07%** | **93** |

That is roughly **two thirds fewer mistakes**, because convolution layers understand which pixels are neighbours.

<p align="center">
  <img src="results/training_curves.png" width="85%" alt="Training curves"/>
</p>

### Per-digit performance (CNN)

| Digit | Precision | Recall | F1-score | Test images |
|---|---|---|---|---|
| 0 | 0.9909 | 0.9969 | 0.9939 | 980 |
| 1 | 0.9913 | 0.9982 | 0.9947 | 1135 |
| 2 | 0.9922 | 0.9903 | 0.9913 | 1032 |
| 3 | 0.9931 | 0.9921 | 0.9926 | 1010 |
| 4 | 0.9879 | 0.9969 | 0.9924 | 982 |
| 5 | 0.9822 | 0.9888 | 0.9855 | 892 |
| 6 | 0.9979 | 0.9823 | 0.9900 | 958 |
| 7 | 0.9874 | 0.9922 | 0.9898 | 1028 |
| 8 | 0.9887 | 0.9897 | 0.9892 | 974 |
| 9 | 0.9950 | 0.9782 | 0.9865 | 1009 |

Overall: accuracy **0.9907**, macro-average precision, recall and F1 all **0.9906**.

- The easiest digits are **1** and **0** (F1 of 0.9947 and 0.9939)
- The hardest are **5** (F1 0.9855) and **9** (F1 0.9865), and 9 has the lowest recall (0.9782)

<p align="center">
  <img src="results/confusion_matrix.png" width="48%" alt="Confusion matrix"/>
  <img src="results/misclassified.png" width="48%" alt="Misclassified images"/>
</p>

The right picture shows images the CNN got wrong. Many of them are hard even for a person to read.

## 🖊️ Try it with your own handwriting

**Draw with your mouse** (needs the saved model in `model/`):

```bash
python draw_and_predict.py
```

A window opens. Draw a digit, press **Predict**, and the model shows its guess with a confidence percentage and the top 3 candidates.

**Predict from a photo or image file**, for example the included sample:

```bash
python predict_digit.py samples/sample_image.jpeg
```

<p align="center">
  <img src="samples/sample_image.jpeg" height="260" alt="Sample handwritten digit photo"/>
</p>

Real photos are different from MNIST, so `digit_utils.py` prepares them the same way MNIST was prepared: it removes shadows and uneven light, thickens thin pen lines, crops the digit, fits it into a 20 x 20 box and centres it on a 28 x 28 canvas.

## 🚀 How to run

```bash
# 1. Clone the repository
git clone https://github.com/Prathmesh2007-16/CodeAlpha_Handwritten_Character_Recognition.git
cd CodeAlpha_Handwritten_Character_Recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional quick check of your setup (about 1 minute)
python handwritten_recognition.py --quick

# 4. Full training (about 7 minutes on a normal laptop CPU)
python handwritten_recognition.py
```

Charts and tables are saved in `results/`, and the trained model is saved as `model/digit_cnn.keras`.

## 🗂️ Project structure

```
CodeAlpha_Handwritten_Character_Recognition/
├── handwritten_recognition.py   # data, baseline, CNN, training and evaluation
├── digit_utils.py               # turns any picture of a digit into the 28x28 MNIST format
├── draw_and_predict.py          # draw a digit with the mouse and predict it
├── predict_digit.py             # predict the digit in an image file
├── model/
│   └── digit_cnn.keras          # the trained CNN
├── results/                     # charts and tables created by the training script
├── samples/
│   └── sample_image.jpeg        # a photo of a handwritten digit to try
├── requirements.txt
└── README.md
```

## 🧠 Key concepts in simple words

- **CNN:** a neural network built for images. Small filters slide over the picture and learn to spot strokes, curves and corners
- **Epoch:** one full pass of the model over all training images
- **Dropout:** randomly switching off some neurons during training so the model does not just memorise the examples
- **Early stopping:** ending training once the validation loss stops improving, which prevents overfitting
- **Softmax:** turns the final numbers into probabilities that add up to 100%
- **Baseline:** a simple reference model, used to prove that the CNN really adds value

## ⚠️ Limitations

- MNIST digits are clean, centred and size-normalised, so accuracy on messy real-world handwriting will usually be lower than 99%
- The model only knows the digits 0 to 9. Recognising letters would need a dataset such as EMNIST
- Training ran on a laptop CPU, so the network is intentionally small

## 👨‍💻 Author

**Prathmesh Chaure**, 3rd-year Electronics (VLSI Design & Technology) student exploring VLSI, embedded systems and machine learning.

[![Portfolio](https://img.shields.io/badge/Portfolio-7c5cff?style=flat-square&logo=vercel&logoColor=white)](https://portfolio-five-lemon-47.vercel.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0a66c2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prathmesh-chaure-b38612335)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Prathmesh2007-16)

*Built during the CodeAlpha Machine Learning internship.*