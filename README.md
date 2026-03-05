# Automated Medical Image Classification Pipeline (DermaMNIST)

## Overview
This project implements an automated deep learning pipeline for medical image classification using the DermaMNIST dataset. The objective is to build a reproducible workflow that performs image ingestion, preprocessing, augmentation, model training, and inference for dermatological image classification.

The pipeline is designed to simulate a real-world medical imaging workflow where incoming image data can be automatically processed and classified using a trained convolutional neural network (CNN).

---

## Dataset

**Dataset:** DermaMNIST  
**Source:** MedMNIST Benchmark Dataset

DermaMNIST is a dermatology image dataset consisting of dermatoscopic images representing multiple skin disease categories.

**Dataset Characteristics**

- Image size: 28 × 28 RGB images  
- Classes: Multiple dermatological disease categories  
- Dataset type: Multi-class classification  
- Total samples: ~10,000 images across training, validation, and test sets

The dataset provides a standardized benchmark for evaluating medical image classification algorithms.

---

## Pipeline Architecture

The project implements an automated pipeline consisting of the following stages:

1. **Data Ingestion**
   - Automatic loading of new image data
   - Directory-based dataset organization

2. **Image Preprocessing**
   - Image resizing and normalization
   - Conversion to tensor format for model compatibility

3. **Data Augmentation**
   - Random image transformations to improve model generalization
   - Techniques include rotation, flipping, and normalization

4. **Model Training**
   - Convolutional Neural Network (CNN) architecture implemented in PyTorch
   - Training performed on labeled dermatological image data

5. **Model Evaluation**
   - Performance measured using accuracy and classification metrics
   - Evaluation performed on a held-out test dataset

6. **Inference Pipeline**
   - Automatic prediction generation for new images
   - Model outputs predicted dermatological class labels

---

## Model Architecture

A Convolutional Neural Network (CNN) was implemented for multi-class image classification.

The model architecture includes:

- Convolutional layers for feature extraction
- Non-linear activation functions
- Pooling layers for dimensionality reduction
- Fully connected layers for classification

CNNs are well-suited for medical imaging tasks due to their ability to learn spatial feature hierarchies from image data.

---

## Automation Workflow

The pipeline includes a file monitoring mechanism that detects newly added images and triggers preprocessing and inference steps automatically.

This simulates a production-like workflow where medical images can be continuously processed without manual intervention.

Automation was implemented using:

- **Watchdog** for file monitoring
- Automated preprocessing and prediction scripts

---

## Model Performance

The trained CNN model achieved approximately:

- **Accuracy:** ~85% on the test dataset

Performance indicates that convolutional architectures can effectively learn patterns in dermatological images even with relatively small image sizes.

---

## Technologies Used

### Programming
- Python

### Deep Learning
- PyTorch
- Torchvision

### Data Processing
- NumPy
- Pandas

### Image Processing
- PIL (Python Imaging Library)

### Automation
- Watchdog


