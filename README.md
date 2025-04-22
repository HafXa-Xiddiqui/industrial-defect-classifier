# Industrial Defect Classifier 🔍🛠

A deep learning project for detecting and classifying surface defects in industrial components using the NEU Surface Defect Database and EfficientNet-based CNNs.

## 🚀 Overview

This project uses convolutional neural networks (CNNs) to classify grayscale images of industrial defects into 6 categories. The model achieves up to 95% validation accuracy using data augmentation, transfer learning, and regularization techniques.

Dataset: [NEU Surface Defect Database on Kaggle](https://www.kaggle.com/datasets/rdsunday/neu-urface-defect-database)

## 🧠 Model

- Architecture: EfficientNetB0 (pretrained on ImageNet)
- Input shape: 200x200 RGB
- Training accuracy: ~92%
- Validation accuracy: ~95%
- Framework: TensorFlow / Keras

## 📊 Results

- Validation Accuracy: **95.28%**
- Techniques Used: Data Augmentation, Dropout, L2 Regularization, Batch Normalization

## 📁 Project Structure

- `scripts/train_model.py` — training pipeline
- `models/` — saved Keras model
- `notebooks/` — Jupyter notebooks for visualization
- `requirements.txt` — all dependencies
- `README.md` — you are here!

## 📌 Skills Applied

- Computer Vision
- Deep Learning (CNNs)
- Image Preprocessing
- Transfer Learning
- TensorFlow & Keras
- Model Evaluation & Deployment Prep


