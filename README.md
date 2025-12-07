# MNIST Digit Prediction

This repository contains a simple deep learning project for handwritten digit recognition using the MNIST dataset. The project features an improved training process with enhanced data augmentation and a user-friendly graphical interface that allows drawing digits with an adjustable brush size.

## Features

- **Improved Training Process**:  
  - Enhanced data augmentation techniques including random rotations and affine transformations.
  - Optimizer switch to Adam and a dynamic learning rate scheduler (ReduceLROnPlateau) for efficient training.
  - Early stopping mechanism to save the best performing model.

- **Interactive GUIs**:  
  - Web: Flask-served HTML canvas at `/` with prediction and conv-layer activation previews.
  - Desktop: Tkinter drawing interface with adjustable brush size and live predictions.

## Structure

- **train.py**: Script for training the convolutional neural network (CNN) on the MNIST dataset.
- **model.py**: Defines the `SimpleCNN` architecture.
- **utils.py**: Utility functions for preprocessing images (centering and normalization).
- **main.py**: Provides the drawing interface and digit prediction using the trained model.

## Data
- MNIST dataset (handwritten digits) — download via Kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## Getting Started

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/ksavill/MNIST-Digit-Prediction.git
   cd mnist-digit-prediction
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**  
   ```bash
   python train.py
   ```

4. **Run the Web UI (HTML canvas at `/`)**  
   ```bash
   python app.py
   ```
   Then open http://localhost:8000 and draw a digit on the canvas to see predictions. Use the “Reload model” button after retraining to pick up new weights.

5. **(Optional) Run the original Tkinter GUI**  
   ```bash
   python main.py
   ```

## Usage Recap

- Install: `pip install -r requirements.txt`
- Train: `python train.py` (best weights -> `model_best.pth`, final -> `model_final.pth`)
- Run web UI: `python app.py`, open http://localhost:8000
- After retraining: click “Reload model” in the web UI to load the new `model_best.pth` without restarting
- Optional desktop UI: `python main.py`

## Using uv

If you prefer uv for environment management:

1. Install uv (see https://github.com/astral-sh/uv)
2. Sync deps and create a virtual env:
   ```bash
   uv sync
   ```
3. Run the app:
   ```bash
   uv run app.py
   ```
4. Train (optional):
   ```bash
   uv run train.py
   ```

uv uses `pyproject.toml` in this repo; the virtual env is managed automatically under `.venv` by default.
