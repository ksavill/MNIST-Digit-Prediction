# MNIST Digit Prediction

This repository contains a simple deep learning project for handwritten digit recognition using the MNIST dataset. The project features an improved training process with enhanced data augmentation and a user-friendly graphical interface that allows drawing digits with an adjustable brush size.

## Features

- **Improved Training Process**:  
  - Enhanced data augmentation techniques including random rotations and affine transformations.
  - Optimizer switch to Adam and a dynamic learning rate scheduler (ReduceLROnPlateau) for efficient training.
  - Early stopping mechanism to save the best performing model.

- **Interactive GUI**:  
  - A Tkinter-based drawing interface where you can paint digits on a 28x28 grid.
  - A slider control to adjust brush size dynamically.
  - Real-time digit prediction with visualization of intermediate convolutional layer activations.

## Structure

- **train.py**: Script for training the convolutional neural network (CNN) on the MNIST dataset.
- **model.py**: Defines the `SimpleCNN` architecture.
- **utils.py**: Utility functions for preprocessing images (centering and normalization).
- **main.py**: Provides the drawing interface and digit prediction using the trained model.

## Getting Started

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/ksavill/MNIST-Digit-Prediction.git
   cd mnist-digit-prediction

2. **Install Depdnencies**
    ```bash
    pip install torch torchvision matplotlib pillow

3. **Train the Model**
   ```bash
   python train.py

4. **Run the GUI**
5. ```bash
   python main.py
