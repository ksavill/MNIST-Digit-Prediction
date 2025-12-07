# MNIST Digit Prediction - Architecture Overview

This document outlines how the application trains and serves a convolutional neural network (CNN) to recognize handwritten digits, and how the web UI interacts with the model.

## Components

- **Model** (`model.py`): A small CNN (`SimpleCNN`) with two padded conv+BN+ReLU blocks, dropout, and two fully connected layers producing logits for 10 classes.
- **Training** (`train.py` or web UI `app.py` training endpoints):
  - Dataset: MNIST (28×28 grayscale digits).
  - Augmentation: rotation, affine jitter, random erasing.
  - Regularization: weight decay, label smoothing, dropout.
  - Scheduler: ReduceLROnPlateau on validation loss, early stopping.
  - Checkpoints: best (`model_best.pth`), final (`model_final.pth`).
- **Preprocessing** (`utils.py`):
  - Centers the drawn digit via bounding box.
  - Inverts to match MNIST polarity (white digit on black background).
  - Resizes to 28×28, normalizes to MNIST mean/std.
- **Inference** (`app.py`):
  - Loads `model_best.pth` on startup (or after web training completes/reload).
  - Exposes `/predict` to accept canvas PNG (base64), preprocess, run model, return digit, confidence, top-3, and activation previews.
- **Web UI** (`templates/index.html`):
  - Canvas drawing, adjustable brush, predict/reset/reload model.
  - Training control: start/cancel training, live status, CUDA badge, loss/accuracy history, activations display.
  - README and this architecture doc rendered inline for onboarding.

## Data Flow (Prediction)

1. User draws on HTML canvas (default brush size 12).
2. Canvas PNG is sent to `/predict` as data URL.
3. Backend decodes, centers, inverts, resizes, normalizes -> tensor.
4. Model forward pass returns logits; softmax used for probabilities.
5. Response includes digit, confidence, top-3, and conv1/conv2 feature map thumbnails.

## Data Flow (Training)

1. Training can be run via CLI (`python train.py`) or started in-browser (calls `/train/start`).
2. MNIST is downloaded (or reused) under `data/`. Train/val split uses 10% for validation.
3. Each epoch: augment -> forward -> loss (CrossEntropy with label smoothing) -> backprop (Adam).
4. Scheduler steps on val loss; best checkpoint saved when val loss improves; early stopping after patience.
5. On completion, `model_best.pth` is reloaded for inference; history remains visible in the UI.

## Serving & Reloading

- The Flask app loads `model_best.pth` at startup. If missing, predict will ask you to train first.
- After training (CLI or web), click “Reload model” in the UI to pull in new weights (web training auto-loads at end).

## Troubleshooting Accuracy

- Ensure preprocessing inversion is used (matches MNIST polarity).
- Retrain after preprocessing changes; reload the model.
- Check CUDA availability (UI badge). If “no” on a CUDA-capable machine, install a CUDA-enabled torch wheel.
- If overfitting, consider reducing augmentation intensity (e.g., lower RandomErasing p/scale) or lowering epochs.

## Files to Know

- `model.py` — defines `SimpleCNN`
- `utils.py` — centering, inversion, normalization
- `train.py` — standalone training loop (CLI)
- `app.py` — Flask server, predict/train endpoints, activation hooks, README/doc serving
- `templates/index.html` — UI for drawing, predictions, training, docs


