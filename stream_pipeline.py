# stream_pipeline.py
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import mlflow

from utils import (
    PROCESSED_PATH,
    download_dermamnist,
    validate_images,
    preprocess_and_augment
)

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class DermMNISTDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")

        # Use regex to parse label safely
        match = re.search(r'_label_(\d+)', img_name)
        if not match:
            raise ValueError(f"Cannot parse label from filename: {img_name}")
        label = int(match.group(1))

        if self.transform:
            img = self.transform(img)

        return img, label

def train_model():
    print("\n Starting training ...")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = DermMNISTDataset(PROCESSED_PATH, transform)
    if len(dataset) < 10:
        print(" Not enough data.")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_experiment("DermaMNIST_Streaming")
    with mlflow.start_run():
        for epoch in range(10):
            model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, preds = torch.max(model(images), 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f" Accuracy: {acc:.2f}%")
        torch.save(model.state_dict(), os.path.join(PROCESSED_PATH, "dermamnist_cnn_stream.pth"))

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".png"):
            print(f" New file detected: {event.src_path}")
            train_model()

if __name__ == "__main__":
    print("\n============== DermaMNIST Streaming Pipeline ==============")
    download_dermamnist()
    X, y, invalid = validate_images(npz_file="dermamnist.npz", split="train")
    preprocess_and_augment(X, y, invalid_indices=invalid)
    train_model()

    print(f" Watching for changes in: {PROCESSED_PATH}")
    observer = Observer()
    observer.schedule(NewFileHandler(), PROCESSED_PATH, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
