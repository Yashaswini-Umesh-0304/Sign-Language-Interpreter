# Import necessary modules
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe's hand detection utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure MediaPipe Hands for static image processing
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the directory where the dataset is stored
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []
labels = []

EXPECTED_LANDMARKS = 21  # Number of landmarks expected for a single hand

# Function to resize image with padding to preserve aspect ratio
def resize_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)  # Determine scaling factor
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_size[1] - new_w
    pad_h = target_size[0] - new_h
    padded = cv2.copyMakeBorder(
        resized,
        pad_h // 2, pad_h - pad_h // 2,  # Top and bottom padding
        pad_w // 2, pad_w - pad_w // 2,  # Left and right padding
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # White background
    )
    return padded

# Process each class directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the current class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [0] * (EXPECTED_LANDMARKS * 2)

        # Read the image and resize with padding
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Error reading image: {dir_}/{img_path}")
            continue
        img_padded = resize_with_padding(img)
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)

        # Check for detected landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_ = [], []
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # Calculate bounding box dimensions
                min_x, max_x = min(x_), max(x_)
                min_y, max_y = min(y_), max(y_)
                width = max_x - min_x
                height = max_y - min_y

                # Normalize coordinates relative to the bounding box
                for i, lm in enumerate(hand_landmarks.landmark):
                    data_aux[i * 2] = (lm.x - min_x) / width if width > 0 else 0
                    data_aux[i * 2 + 1] = (lm.y - min_y) / height if height > 0 else 0

            # Append processed data and labels
            data.append(data_aux)
            labels.append(dir_)
        else:
            # Print message for images with no detected landmarks
            print(f"No landmarks detected in image: {dir_}/{img_path}")

# Serialize and save processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
