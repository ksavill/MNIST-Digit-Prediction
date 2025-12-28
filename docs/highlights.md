# MNIST Digit Prediction — Highlights

## 1. Setup (Zero Friction)

**Dependencies** — Just two commands to get running:

```bash
git clone https://github.com/ksavill/MNIST-Digit-Prediction.git
cd MNIST-Digit-Prediction
pip install -r requirements.txt   # or: uv sync
```

**Automatic Data Download** — The MNIST dataset (70,000 handwritten digit images) downloads itself the first time you train. No manual download required.

```python
# From train.py / app.py — torchvision handles everything:
datasets.MNIST('data/', train=True, download=True, ...)
```

This fetches ~11 MB of compressed data into `data/MNIST/raw/` and extracts it automatically.

---

## 2. ML Architecture (SimpleCNN)

A lightweight but effective Convolutional Neural Network designed for 28×28 grayscale images.

```
Input (1×28×28)
    │
    ▼
┌─────────────────────────────┐
│ Conv1: 5×5, 32 filters      │  → BatchNorm → ReLU → MaxPool(2×2)
└─────────────────────────────┘
    │  Output: 32×14×14
    ▼
┌─────────────────────────────┐
│ Conv2: 5×5, 64 filters      │  → BatchNorm → ReLU → MaxPool(2×2)
└─────────────────────────────┘
    │  Output: 64×7×7 → Dropout(25%)
    ▼
┌─────────────────────────────┐
│ Flatten → FC(3136→128)      │  → ReLU → Dropout(50%)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ FC(128→10) → Logits         │  (one output per digit 0-9)
└─────────────────────────────┘
```

**Key Design Choices:**
- **Batch Normalization** — Faster, more stable training
- **Dropout (25% + 50%)** — Prevents overfitting on the small dataset
- **Label Smoothing (0.1)** — Stops the model from being overconfident

**Training Enhancements:**
- Data augmentation (rotation, affine transforms, random erasing)
- Adam optimizer with learning rate scheduler
- Early stopping (saves best model automatically)

---

## 3. Using It

### Train the Model
```bash
python train.py
```
- Downloads MNIST automatically on first run
- Saves `model_best.pth` (best validation) and `model_final.pth`
- Typically reaches ~99% accuracy in under 10 epochs

### Run the Web UI
```bash
python app.py
# Open http://localhost:8000
```

**Web UI Features:**
- Draw a digit on the canvas with your mouse
- See live predictions with confidence scores
- Visualize what the CNN "sees" (conv layer activations)
- Train/retrain directly from the browser
- Reload model weights without restarting the server

### How Prediction Works
1. Your drawing is captured as a base64 image
2. Preprocessed: resized to 28×28, centered, normalized
3. Fed through the CNN → outputs 10 logits
4. Softmax converts to probabilities → top prediction shown

---

## Quick Reference

| Command | What it does |
|---------|--------------|
| `pip install -r requirements.txt` | Install dependencies |
| `python train.py` | Train model (auto-downloads data) |
| `python app.py` | Launch web UI on port 8000 |
| `python main.py` | (Optional) Desktop Tkinter UI |

**Files to Know:**
- `model.py` — CNN architecture
- `train.py` — Training script
- `app.py` — Flask web server + in-browser training
- `model_best.pth` — Trained weights (loadable)

