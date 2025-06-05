# image-classification

# CNN Image Classification App

This repository contains a simple image classification web app built with **Streamlit** and **PyTorch**. The app uses a Convolutional Neural Network (CNN) to classify grayscale images (32x32) into one of 10 classes, similar to CIFAR-10 categories.

## Features

- Upload a grayscale image (32x32 pixels) in PNG or JPG format.
- The app predicts the class using a trained CNN model.
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

## Files

- `app.py` — Streamlit web app for image classification.
- `complex_cnn_model_state_dict.pth` — Trained PyTorch model weights.
- `complex_cnn_model.pth` — (Optional) Full model checkpoint.
- `img-classification.ipynb` — Jupyter notebook for data prep, training, and evaluation.
- `requirements.txt` — Python dependencies.
- `standard_scaler_model.pkl` — Scaler used for preprocessing (if needed).

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/antonychackotc/cnn-image-classification.git
   cd cnn-image-classification
