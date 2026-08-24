# 🚦 Traffic Sign Recognition Using CNN

A deep learning project that recognizes traffic signs using a **Convolutional Neural Network (CNN)** trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.

## 📌 Overview

The system takes a traffic sign image as input and predicts the corresponding traffic sign class along with its confidence score.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* CNN

## 📊 Dataset

* **43 traffic sign classes**
* **26,640 images**
* Image size: **32 × 32 × 3**
* Training images: **21,312**
* Validation images: **5,328**

## 🧠 Model

The project uses a CNN with multiple convolution, pooling, dense, and dropout layers for traffic sign classification.

### Results

* **Training Accuracy:** 98.67%
* **Validation Accuracy:** 99.36%

## 📂 Project Structure

```text
Traffic-Sign-Recognition/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── test_images/
│   └── man-at-work.jpg
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Preprocess the dataset:

```bash
python src/preprocessing.py
```

Train the model:

```bash
python src/train.py
```

Predict a traffic sign:

```bash
python src/predict.py
```

The model can also be tested with a **custom traffic sign image**.

## 🎯 Example

```text
Predicted Traffic Sign: General caution
Confidence: 64.97%
```

## 🚀 Future Improvements

* Real-time traffic sign recognition
* Web-based interface
* Data augmentation
* Improved performance on real-world images
* Camera-based prediction
