import random
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import SimpleCNN


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, device, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch {epoch} "
                f"[{batch_idx * len(data)}/{len(loader.dataset)}] "
                f"Loss: {loss.item():.4f}"
            )
    return total_loss / len(loader.dataset)


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


def main():
    seed_everything(42)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    common_loader_kwargs = {
        "num_workers": 2 if use_cuda else 0,
        "pin_memory": use_cuda,
    }

    batch_size = 64
    test_batch_size = 1000

    # -----------
    # Data + augmentations
    # -----------
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

    full_train = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST('../data', train=False, transform=test_transform)

    # Train/val split from training set
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **common_loader_kwargs)

    model = SimpleCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    num_epochs = 10
    best_val_loss = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "model_best.pth")
            print(f"New best model saved at epoch {epoch} (val_loss={val_loss:.4f}).")
        else:
            no_improve += 1

        print(
            f"=== Epoch {epoch} === "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Final evaluation on the held-out test set
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    torch.save(model.state_dict(), "model_final.pth")
    print(f"Training complete. Final model saved to model_final.pth")
    print(f"Test set: loss={test_loss:.4f} acc={test_acc:.2f}%")


if __name__ == '__main__':
    main()