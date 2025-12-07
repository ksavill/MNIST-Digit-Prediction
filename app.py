import base64
import io
import numpy as np
from pathlib import Path
from typing import Dict
import threading
import time

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import markdown as md

from model import SimpleCNN
from utils import preprocess_image

app = Flask(__name__, static_folder="static", template_folder="templates")
ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.md"
DATA_PATH = ROOT / "data"

training_state = {
    "running": False,
    "epoch": 0,
    "num_epochs": 0,
    "step": 0,
    "steps_per_epoch": 0,
    "train_loss_avg": None,
    "train_loss_epoch": None,
    "val_loss": None,
    "val_acc": None,
    "history": {"train_loss": [], "val_loss": [], "val_acc": []},
    "message": "",
    "error": None,
    "last_update": None,
    "cuda": torch.cuda.is_available(),
}
state_lock = threading.Lock()
model_loaded = False
train_thread: threading.Thread | None = None
stop_event: threading.Event | None = None

def load_model() -> SimpleCNN:
    """Load the trained model once at startup."""
    model = SimpleCNN()
    global model_loaded
    try:
        state = torch.load(ROOT / "model_best.pth", map_location="cpu")
        model.load_state_dict(state)
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    model.eval()
    return model


model: SimpleCNN = load_model()


def data_url_to_pil(data_url):
    """Convert a base64 data URL image into a BytesIO buffer for PIL."""
    if "," not in data_url:
        raise ValueError("Invalid data URL.")
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return io.BytesIO(image_bytes)


def decode_image(data_url):
    """Decode data URL to a grayscale PIL.Image."""
    image_buffer = data_url_to_pil(data_url)
    return Image.open(image_buffer).convert("L")


def tensor_maps_to_dataurls(tensor, limit=8, size=84):
    """
    Convert the first `limit` feature maps in a BCHW tensor to base64 PNG data URLs.
    """
    arr = tensor.squeeze(0).detach().cpu().numpy()
    urls = []
    for i in range(min(limit, arr.shape[0])):
        fmap = arr[i]
        norm = fmap - fmap.min()
        denom = norm.max() if norm.max() > 0 else 1e-6
        norm = norm / denom
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize(
            (size, size), Image.BILINEAR
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        urls.append(f"data:image/png;base64,{b64}")
    return urls


def read_readme_text() -> str:
    """Return README contents as text; empty string on failure."""
    try:
        return README_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "model_not_loaded", "detail": "Train a model first."}), 503
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    data_url = (payload or {}).get("image")
    if not data_url:
        return jsonify({"error": "missing_image"}), 400

    try:
        pil_image = decode_image(data_url)
    except Exception as exc:
        return jsonify({"error": "bad_image", "detail": str(exc)}), 400

    img_tensor = preprocess_image(pil_image)

    # Capture activations with forward hooks so we don't run two passes.
    activations = {}

    def capture(name):
        def hook(_, __, output):
            activations[name] = output
        return hook

    hooks = [
        model.conv1.register_forward_hook(capture("conv1")),
        model.conv2.register_forward_hook(capture("conv2")),
    ]

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        top_prob, pred_idx = torch.max(probs, dim=0)
        top3_prob, top3_idx = torch.topk(probs, 3)

    for h in hooks:
        h.remove()

    response = {
        "digit": int(pred_idx),
        "confidence": float(top_prob),
        "top3": [
            {"digit": int(idx), "prob": float(prob)}
            for prob, idx in zip(top3_prob, top3_idx)
        ],
        "activations": {
            "conv1": tensor_maps_to_dataurls(activations.get("conv1", torch.empty(0))),
            "conv2": tensor_maps_to_dataurls(activations.get("conv2", torch.empty(0))),
        },
    }
    return jsonify(response)


@app.route("/reload", methods=["POST"])
def reload():
    """Reload model weights from disk (e.g., after retraining)."""
    global model, model_loaded
    try:
        model = load_model()
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500
    return jsonify({"status": "ok", "model_loaded": model_loaded})


@app.route("/readme", methods=["GET"])
def readme():
    """Return README contents so the UI can display setup instructions."""
    text = read_readme_text()
    if not text:
        return jsonify({"error": "readme_not_found"}), 404
    html = md.markdown(text, extensions=["fenced_code", "tables"])
    return jsonify({"content": text, "html": html})


@app.route("/doc/architecture", methods=["GET"])
def doc_architecture():
    path = ROOT / "docs" / "architecture.md"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return jsonify({"error": "doc_not_found"}), 404
    html = md.markdown(text, extensions=["fenced_code", "tables"])
    return jsonify({"content": text, "html": html})


def build_loaders(batch_size=64, test_batch_size=1000, val_split=0.1, device=None):
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(str(DATA_PATH), train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(str(DATA_PATH), train=False, download=True, transform=test_transform)

    val_size = int(val_split * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    use_cuda = device and device.type == "cuda"
    common_loader_kwargs = {"num_workers": 2 if use_cuda else 0, "pin_memory": use_cuda}

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader, test_loader


def evaluate(model, device, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return avg_loss, accuracy


def train_background(num_epochs=5, lr=0.001, seed=42, stop_event: threading.Event | None = None):
    global model, model_loaded
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    train_loader, val_loader, _ = build_loaders(device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    with state_lock:
        training_state.update({
            "running": True,
            "epoch": 0,
            "num_epochs": num_epochs,
            "step": 0,
            "steps_per_epoch": len(train_loader),
            "train_loss_avg": None,
            "train_loss_epoch": None,
            "val_loss": None,
            "val_acc": None,
            "history": {"train_loss": [], "val_loss": [], "val_acc": []},
            "message": "Training started",
            "error": None,
            "last_update": time.time(),
            "cuda": device.type == "cuda",
        })

    best_val = float("inf")
    patience = 3
    no_improve = 0

    try:
        for epoch in range(1, num_epochs + 1):
            print(f"[train] Epoch {epoch}/{num_epochs} starting...")
            model.train()
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader, start=1):
                if stop_event and stop_event.is_set():
                    raise RuntimeError("Training cancelled")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.size(0)

                if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                    print(
                        f"[train] Epoch {epoch}/{num_epochs} "
                        f"batch {batch_idx}/{len(train_loader)} "
                        f"loss={loss.item():.4f}"
                    )

                with state_lock:
                    training_state.update({
                        "epoch": epoch,
                        "step": batch_idx,
                        "train_loss_avg": loss.item(),
                        "last_update": time.time(),
                        "message": f"Epoch {epoch}/{num_epochs} batch {batch_idx}/{len(train_loader)}",
                    })

            avg_train_loss = total_loss / len(train_loader.dataset)
            val_loss, val_acc = evaluate(model, device, val_loader, criterion)
            scheduler.step(val_loss)
            print(
                f"[train] Epoch {epoch} done. "
                f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
            )

            with state_lock:
                training_state.update({
                    "train_loss_epoch": avg_train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "history": {
                        "train_loss": training_state["history"]["train_loss"] + [avg_train_loss],
                        "val_loss": training_state["history"]["val_loss"] + [val_loss],
                        "val_acc": training_state["history"]["val_acc"] + [val_acc],
                    },
                    "last_update": time.time(),
                    "message": f"Epoch {epoch} complete",
                })

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(model.state_dict(), ROOT / "model_best.pth")
            else:
                no_improve += 1

            if no_improve >= patience:
                with state_lock:
                    training_state.update({
                        "message": "Early stopping (no val improvement)",
                        "last_update": time.time(),
                    })
                break

        torch.save(model.state_dict(), ROOT / "model_final.pth")
        model_loaded = True
        model = load_model()
        with state_lock:
            history = {
                "train_loss": training_state["history"]["train_loss"] + [avg_train_loss],
                "val_loss": training_state["history"]["val_loss"] + [val_loss],
                "val_acc": training_state["history"]["val_acc"] + [val_acc],
            }
            training_state.update({
                "running": False,
                "message": "Training complete",
                "last_update": time.time(),
                "history": history,
            })
    except Exception as exc:
        msg = "Training failed" if not (stop_event and stop_event.is_set()) else "Training cancelled"
        print(f"[train] {msg}: {exc}")
        with state_lock:
            training_state.update({
                "running": False,
                "error": str(exc),
                "message": msg,
                "last_update": time.time(),
            })
            # Preserve history even on failure
            training_state["history"] = {
                "train_loss": training_state["history"]["train_loss"],
                "val_loss": training_state["history"]["val_loss"],
                "val_acc": training_state["history"]["val_acc"],
            }


@app.route("/train/start", methods=["POST"])
def train_start():
    global stop_event, train_thread
    with state_lock:
        if training_state.get("running"):
            return jsonify({"status": "already_running"}), 409
        stop_event = threading.Event()
        # Optimistically mark running to block concurrent starts before the worker sets state.
        training_state.update({
            "running": True,
            "message": "Launching training...",
            "last_update": time.time(),
        })
        # Keep existing history so UI can show last results until new ones arrive.
        # Do not clear history here.

    try:
        payload = request.get_json(force=True, silent=True) or {}
        num_epochs = int(payload.get("epochs", 5))
        lr = float(payload.get("lr", 0.001))
    except Exception:
        return jsonify({"error": "bad_request"}), 400

    try:
        train_thread = threading.Thread(
            target=train_background,
            kwargs={"num_epochs": num_epochs, "lr": lr, "stop_event": stop_event},
            daemon=True,
        )
        train_thread.start()
    except Exception:
        with state_lock:
            training_state.update({"running": False, "message": "Failed to launch training"})
        raise

    return jsonify({"status": "started", "epochs": num_epochs, "lr": lr})


@app.route("/train/status", methods=["GET"])
def train_status():
    with state_lock:
        state = dict(training_state)
        state["cuda"] = torch.cuda.is_available()
        return jsonify(state)


@app.route("/train/cancel", methods=["POST"])
def train_cancel():
    global stop_event
    with state_lock:
        if not training_state.get("running"):
            return jsonify({"status": "not_running"})
        if stop_event:
            stop_event.set()
    return jsonify({"status": "cancelling"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

