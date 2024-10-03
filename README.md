# Face Mark Vision

## Overview

This project demonstrates how to detect face landmarks using machine learning. Face marks are key points on a face, such as the corners of the eyes, tip of the nose, or mouth, used to project filters (like those on Snapchat). In this project, we use the **DLIB dataset** and **PyTorch** for detecting and predicting these face landmarks.

The model architecture is built on **ResNet18**, and the dataset is split for training and validation, using a custom dataset class and transformation pipeline. The model is trained using the **Mean Squared Error (MSE)** loss function.

---

## Requirements

Make sure to install the following libraries before running the code:

```bash
pip install torch torchvision opencv-python matplotlib scikit-image Pillow imutils
```

---

## Project Structure

1. **Import Libraries**: All necessary libraries such as PyTorch, OpenCV, and others are imported to handle image transformations, model building, and dataset handling.
2. **Download Dataset**: The DLIB dataset is used for face landmark detection. It contains over 6666 images of varying dimensions.
3. **Data Visualization**: A simple function to visualize sample images and the corresponding landmarks.
4. **Custom Dataset Class**: A custom dataset class is created to handle image transformations, resizing, and landmarks cropping.
5. **Model Architecture**: We use ResNet18 as the base architecture, with modifications to the input and output layers for landmark prediction.
6. **Training and Validation**: The training loop iterates over the dataset, minimizing the MSE loss, and validates the performance on a separate validation set.
7. **Prediction and Visualization**: After training, the model predicts landmarks on unseen images and compares them with ground truth landmarks.

---

## Code Breakdown

### 1. Import Libraries

```python
import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from skimage import io, transform
import xml.etree.ElementTree as ET

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
```

### 2. Download and Prepare Dataset

Download and extract the dataset:

```python
if not os.path.exists('/content/ibug_300W_large_face_landmark_dataset'):
    !wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
    !tar -xvzf 'ibug_300W_large_face_landmark_dataset.tar.gz'    
    !rm -r 'ibug_300W_large_face_landmark_dataset.tar.gz'
```

### 3. Visualize the Dataset

```python
file = open('ibug_300W_large_face_landmark_dataset/helen/trainset/100032540_1.pts')
points = file.readlines()[3:-1]
landmarks = []

for point in points:
    x, y = point.split(' ')
    landmarks.append([float(x), float(y.strip())])

landmarks = np.array(landmarks)
plt.figure(figsize=(10,10))
plt.imshow(mpimg.imread('ibug_300W_large_face_landmark_dataset/helen/trainset/100032540_1.jpg'))
plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='g')
plt.show()
```

### 4. Custom Dataset and Transformations

Custom Dataset Class for handling images and landmarks:

```python
class FaceLandmarksDataset(Dataset):
    def __init__(self, transform=None):
        tree = ET.parse('ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = 'ibug_300W_large_face_landmark_dataset'

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
            self.crops.append(filename[0].attrib)
            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5
        return image, landmarks
```

### 5. Model Architecture

Using **ResNet18** as the base model, we modify the first and last layers for facial landmarks prediction:

```python
class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
```

### 6. Training and Validation

We use **Adam optimizer** and **MSELoss** for training the model:

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)

for epoch in range(1, num_epochs + 1):
    # Training loop here
```

---

## Running the Project

1. Clone the repository or download the code.
2. Install all the required libraries mentioned in the **Requirements** section.
3. Run the code in sequence for training and predicting face landmarks.

```bash
python face_landmarks_detection.py
```

After training, the model will predict landmarks on test images and display the results.

---

## Results

The model accurately predicts facial landmarks, which can be visualized using Matplotlib.

Example prediction after training:

```python
plt.imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
plt.scatter(predictions[0, :, 0], predictions[0, :, 1], c='r', s=5)
plt.show()
```

---

## References

- [DLIB Dataset](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

