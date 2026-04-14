"""
Train a simple CNN on MNIST and save the model weights.
Run once before building the Docker image, or let the container entrypoint call it
if no weights file is present.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_def import MNISTNet

DATA_DIR = os.environ.get("DATA_DIR", "./data")
MODEL_PATH = os.environ.get("MODEL_PATH", "./model/mnist_cnn.pt")
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "5"))
BATCH_SIZE = 64
LR = 1e-3


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
