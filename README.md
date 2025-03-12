# Facial-Emotion-Detection // ResEmoteNet + ONNX

🚀 This repository is a modified version of [ResEmoteNet](https://github.com/ArnabKumarRoy02/ResEmoteNet), focusing exclusively on the **FER2013 dataset** and contains resources and code for implementing face detection and recognition using OpenCV, Haar cascades, and ONNX runtime inference.

## Overview
This project includes:
- Haar Cascade XML file (`haarcascade_frontalface_default.xml`) for face detection.
- Python notebook (`onnx_inference.ipynb`) demonstrating ONNX model inference.
- Support for handling images for facial recognition tasks.

## Repository Structure
```
├── images
│   ├── person1 (Add images here)
│   └── person2 (Add images here)
├── haarcascade_frontalface_default.xml
├── onnx_inference.ipynb
└── README.md
```

## Usage
### Face Detection
Utilize the provided Haar Cascade (`haarcascade_frontalface_default.xml`) to detect faces in images or video streams with OpenCV.

### ONNX Model Inference
Follow the instructions in `onnx_inference.ipynb` to perform face recognition or other tasks using ONNX models.

## Installation
```bash
pip install opencv-python onnxruntime
```

## License
The face detection XML provided is under the Intel License Agreement for the Open Source Computer Vision Library (OpenCV). See the XML file header for full license details.

## Contributions
Feel free to fork this repository, submit issues, or contribute improvements via pull requests.

---

Developed with ♥ using OpenCV and ONNX Runtime.

