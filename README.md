# Image Classifier using CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model is built using TensorFlow and Keras.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Requirements

- Python 3.7+
- TensorFlow 2.10.0+
- NumPy 1.21.0+
- Matplotlib 3.4.0+

## Installation

Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the image classifier:
```
python image_classifier.py
```

The script will:
1. Load and preprocess the CIFAR-10 dataset
2. Define and train a CNN model
3. Display training progress and accuracy
4. Show sample predictions

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Dense layers for classification
- Dropout for regularization

## Results

The model achieves approximately 70-75% accuracy on the test set after 10 epochs of training.
