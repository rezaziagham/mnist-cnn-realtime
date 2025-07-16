# 🧠 MNIST CNN + Real-Time Digit Recognition

A clean, modular deep learning project for classifying handwritten digits using a Convolutional Neural Network (CNN), and running real-time digit detection using your webcam and OpenCV.

---

## 📌 Features

- ✅ Train a CNN on MNIST dataset
- 📈 Visualize predictions, confusion matrix, and classification metrics
- 📷 Real-time digit detection via webcam
- 🧼 Modular Python structure for clarity and maintainability
- 💾 Trained model saving and reuse

---

## 🗂️ Project Structure

```
mnist_cnn_project/
├── data/                     # Dataset files (MNIST)
│   └── mnist.npz
├── models/                   # Saved trained models
│   └── mnist_cnn_model.h5
├── src/                      # Source code
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── model_builder.py      # CNN architecture
│   ├── train.py              # Training and evaluation logic
│   ├── utils.py              # Visualization and metrics
│   └── realtime_digit_detector.py  # Webcam digit recognition
├── main.py                   # Entry point for training
├── run_realtime.py           # Entry point for real-time detection
├── requirements.txt          # Dependencies
└── README.md                 # Project overview
```

---

## 📥 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rezaziagham/mnist-cnn-realtime.git
cd mnist-cnn-realtime
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Download the MNIST dataset from Keras and place it inside the `data/` folder:

```bash
mkdir data
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz -P data/
```
 or you can use it directly from tensorflow:
 
```bash
from tensorflow.keras.datasets import mnist
```

---

## 🚀 Usage

### 🧠 Train the CNN Model

```bash
python main.py
```

This will:
- Train the model on the MNIST dataset
- Save the model as `models/mnist_cnn_model.h5`

---

### 🎥 Run Real-Time Digit Recognition

```bash
python run_realtime.py
```

This will:
- Launch your webcam
- Detect digits in real-time
- Highlight valid digits with prediction and confidence

> ✅ Press `q` to exit the webcam window

---

## 📊 Example Output

**Training Sample:**

![Training Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)


---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- NumPy, Matplotlib, Seaborn, scikit-learn

Install via:

```bash
pip install -r requirements.txt
```

---

## 📚 Learn More

- [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [OpenCV](https://opencv.org/)

---


## 🛡 License

MIT License - feel free to use, modify, and share.
