# inference_server.py
import os
import time
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import PROCESSED_PATH


# Paths

BASE_DIR = r"D:\DermaMNIST_Pipeline"
INFER_INPUT = os.path.join(BASE_DIR, "inference_input")
INFER_OUTPUT = os.path.join(BASE_DIR, "inference_output")

os.makedirs(INFER_INPUT, exist_ok=True)
os.makedirs(INFER_OUTPUT, exist_ok=True)


# CNN 

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*16*16,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------
# Load trained model
# ---------------------------
MODEL_PATH = os.path.join(PROCESSED_PATH, "dermamnist_cnn_stream.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f" Loaded trained model from {MODEL_PATH}")
else:
    print(" Model not found! Please train using stream_pipeline.py first.")


# Preprocessing

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(image_path):
    # Windows-safe read retry
    for i in range(10):
        try:
            with open(image_path, "rb") as fsrc:
                img = Image.open(fsrc).convert("RGB")
                break
        except PermissionError:
                print(f" File {image_path} is locked. Retrying in 1s...")
                time.sleep(1)
    else:
        raise PermissionError(f" Could not read {image_path} after multiple attempts.")

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    return int(pred.item())


# File Watcher for auto inference

class InferenceHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return
        print(f" New image detected for inference: {event.src_path}")
        try:
            label = predict_image(event.src_path)
            df = pd.DataFrame([[os.path.basename(event.src_path), label]], columns=["filename", "predicted_class"])
            output_csv = os.path.join(INFER_OUTPUT, "predictions.csv")
            if os.path.exists(output_csv):
                df.to_csv(output_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(output_csv, index=False)
            print(f" Prediction saved to {output_csv}")
        except Exception as e:
            print(f"Error processing {event.src_path}: {e}")


# Flask API 
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return jsonify({'predicted_class': int(predicted.item())})



# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Start folder watcher
    event_handler = InferenceHandler()
    observer = Observer()
    observer.schedule(event_handler, path=INFER_INPUT, recursive=False)
    observer.start()
    print(f"Watching folder for new images: {INFER_INPUT}")

    # Start Flask API
    print(" Starting Flask API at http://127.0.0.1:8000/predict")
    try:
        app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
