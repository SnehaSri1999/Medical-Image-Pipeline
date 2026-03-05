# utils.py
from medmnist import DermaMNIST, INFO
from datetime import datetime
import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import albumentations as A

RAW_DATA_PATH = r"D:\DermaMNIST_Pipeline\data\raw"
PROCESSED_PATH = r"D:\DermaMNIST_Pipeline\data\processed"
LOG_PATH = r"D:\DermaMNIST_Pipeline\data\logs"

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

def download_dermamnist():
    print("Downloading DermaMNIST via medmnist...")
    dataset = DermaMNIST(split="train", download=True, root=RAW_DATA_PATH)

    info = INFO.get("dermamnist", {})
    meta = {
        "dataset": info.get("python_class", "DermaMNIST"),
        "description": info.get("description", "No description available"),
        "date_downloaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(dataset)
    }

    log_path = os.path.join(LOG_PATH, "ingestion_log.csv")
    pd.DataFrame([meta]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

    print("Download complete")

def validate_images(npz_file="dermamnist.npz", blur_threshold=30, split="train"):
    data = np.load(os.path.join(RAW_DATA_PATH, npz_file))
    X = data[f"{split}_images"]
    y = data[f"{split}_labels"]

    invalid_indices = []
    for i, img_array in enumerate(X):
        try:
            img = Image.fromarray(img_array)
            img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            if cv2.Laplacian(img_gray, cv2.CV_64F).var() < blur_threshold:
                invalid_indices.append(i)
        except:
            invalid_indices.append(i)

    pd.DataFrame({"invalid_indices": invalid_indices}).to_csv(
        os.path.join(LOG_PATH, "validation_log.csv"), index=False)

    print(f"Validation complete. Invalid images: {len(invalid_indices)}")
    return X, y, invalid_indices

def preprocess_and_augment(X, y, invalid_indices=[], size=(64, 64), augment_count=2):
    transform_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])

    saved_files = []
    for i, (img_array, label) in enumerate(zip(X, y)):
        if i in invalid_indices:
            continue

        img = Image.fromarray(img_array).resize(size)
        filename = f"img_{i}_label_{int(label)}.png"
        img.save(os.path.join(PROCESSED_PATH, filename))
        saved_files.append(filename)

        img_np = np.array(img)
        for j in range(augment_count):
            augmented = transform_aug(image=img_np)["image"]
            filename_aug = f"img_{i}_label_{int(label)}_aug{j}.png"
            Image.fromarray(augmented).save(os.path.join(PROCESSED_PATH, filename_aug))
            saved_files.append(filename_aug)

    print(f"Total processed images saved: {len(saved_files)}")
    return saved_files
