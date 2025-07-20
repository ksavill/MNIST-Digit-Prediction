import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import SimpleCNN

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} "
                  f"[{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)")
    return test_loss, accuracy

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 64, 'shuffle': True}
    test_kwargs = {'batch_size': 1000}

    # -----------
    # Enhanced Data Augmentation
    # -----------
    train_transform = transforms.Compose([
        # Random rotation can help the model generalize better
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Uncomment below if you want to try RandomPerspective:
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Only normalization for the test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = SimpleCNN().to(device)

    # -----------
    # Switch to a different Optimizer (e.g., Adam)
    # -----------
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------
    # Learning Rate Scheduler (ReduceLROnPlateau as an example)
    # -----------
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=2,
                                                     verbose=True)

    num_epochs = 15
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)

        # Step the scheduler with test_loss
        scheduler.step(test_loss)

        # Optional: Simple "Early Stopping" style check
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "model_best.pth")
            print("New best model saved!")

        print(f"=== Finished Epoch {epoch} ===\n")

    # Final save after all epochs
    torch.save(model.state_dict(), "model_final.pth")
    print("Training complete. Final model saved to model_final.pth")

if __name__ == '__main__':
    main()