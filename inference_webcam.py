import cv2
import cv2.data
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from Approach.ResEmoteNet import ResEmoteNet

# Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Emotions labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Load the model
model = ResEmoteNet().to(device)
checkpoint = torch.load('./Weights/fer_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Access the webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (0, 255, 0)  # BGR color
thickness = 1
line_type = cv2.LINE_AA

def detect_emotion(image):
    """Detect emotion from a cropped face image."""
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    return scores

def display_emotions(x, y, w, h, image):
    """Display emotions and their scores on the video frame."""
    crop_img = image[y:y + h, x:x + w]
    pil_crop_img = Image.fromarray(crop_img)
    scores = detect_emotion(pil_crop_img)
    max_index = np.argmax(scores)
    max_emotion = emotions[max_index]

    # Display the max emotion above the bounding box
    org = (x, y - 10)
    cv2.putText(image, max_emotion, org, font, font_scale, font_color, thickness, line_type)

    # Display all emotions and their scores
    org = (x + w + 10, y)
    for idx, emotion in enumerate(emotions):
        text = f"{emotion}: {scores[idx]:.2f}"
        cv2.putText(image, text, (org[0], org[1] + idx * 20), font, font_scale, font_color, thickness, line_type)

def detect_bounding_box(image):
    """Detect faces and display emotions."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display emotions for the detected face
        display_emotions(x, y, w, h, image)

# Main loop for real-time face detection and emotion recognition
print("Press 'q' to quit.")
while True:
    ret, video_frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    detect_bounding_box(video_frame)
    cv2.imshow("ResEmoteNet", video_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
