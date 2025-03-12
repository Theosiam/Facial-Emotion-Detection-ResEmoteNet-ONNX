# ResEmoteNet - Real-Time Facial Emotion Recognition

This repository provides code and resources for real-time face detection and emotion recognition using the ResEmoteNet architecture, ONNX runtime, and OpenCV.

## Overview
ResEmoteNet is a deep convolutional neural network tailored for Facial Emotion Recognition (FER). It uses:
- Residual blocks and Squeeze-and-Excitation (SE) blocks for efficient feature extraction.
- Haar Cascades (`haarcascade_frontalface_default.xml`) for face detection.
- Webcam-based real-time inference scripts leveraging PyTorch and ONNX Runtime.

## Project Contents
- `ResEmoteNet.py`: Definition of the neural network architecture.
- `inference_webcam.py`: Python script for real-time webcam inference using PyTorch.
- `onnx_inf_webcam.py`: Python script demonstrating ONNX model inference on webcam feed.

## Structure
```
├── Weights
│   └── fer_model.pth (FER2013 weights go here)
├── haarcascade_frontalface_default.xml
├── ResEmoteNet.py
├── inference_webcam.py
├── onnx_extrat.ipynb
├── onnx_inf_webcam.py
└── README.md
```

## Installation
```bash
pip install opencv-python torch torchvision onnxruntime numpy pillow
```

## Usage
### Real-Time Webcam Inference
To run emotion recognition from webcam:

- Using PyTorch:
```bash
python inference_webcam.py
```

- Using ONNX:
```bash
python onnx_inf_webcam.py
```

Ensure the `fer_model.pth` weight file is in the `Weights` folder.

## Dependencies
Install the necessary Python packages:
```bash
pip install opencv-python torch torchvision onnxruntime numpy pillow
```

## License
The Haar cascade XML is provided under the Intel License Agreement for Open Source Computer Vision Library (OpenCV). See XML file header for full details.

## Contributions
Your contributions are welcome. Please fork this repository, open issues, or submit pull requests.

---

Crafted with ♥ for seamless emotion recognition using ResEmoteNet and OpenCV.

