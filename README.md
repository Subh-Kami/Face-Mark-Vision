# FaceMarkVision

FaceMarkVision is a deep learning project designed to detect facial landmarks in images using PyTorch. The project leverages the DLIB dataset for training a neural network model based on the ResNet-18 architecture. This model can accurately predict facial landmarks, enabling applications such as augmented reality filters, facial recognition, and more.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Features

- Detects 68 facial landmarks from input images.
- Data augmentation techniques for improved model robustness.
- Easy-to-use interface for training and prediction.
- Visualizes results with predicted and ground truth landmarks.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/FaceLandmarkVision.git
   cd FaceLandmarkVision
   ```

2. Set up a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. Install the required libraries:

   ```bash
   pip install torch torchvision matplotlib opencv-python scikit-image imutils
   ```

4. Download the DLIB dataset:

   ```bash
   wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
   tar -xvzf ibug_300W_large_face_landmark_dataset.tar.gz
   rm -r ibug_300W_large_face_landmark_dataset.tar.gz
   ```

## Dataset

The project utilizes the [IBUG 300W Large Face Landmark Dataset](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz), which contains over 6666 annotated images for training the model.

## Usage

1. To train the model, run:

   ```bash
   python src/train.py
   ```

2. To perform predictions on unseen images, run:

   ```bash
   python src/predict.py
   ```

3. The predicted landmarks will be visualized on the output images.

## Results

The trained model is capable of accurately detecting facial landmarks, as shown in the visualizations provided in the `results` folder.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
