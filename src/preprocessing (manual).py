import cv2
import numpy as np

def preprocess_image(img_path, size=(200, 200), edge=False):
    # Step 1: Load and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 3: (Optional) Add edge information
    if edge:
        edges = cv2.Canny(enhanced, 100, 200)
        # Stack enhanced + edges as 2-channel input
        stacked = np.stack([enhanced, edges], axis=-1)
        # Resize to desired shape
        resized = cv2.resize(stacked, size)
        # Normalize to 0–1
        normalized = resized / 255.0
        return normalized
    else:
        # Resize and normalize just the CLAHE-enhanced grayscale
        resized = cv2.resize(enhanced, size)
        normalized = resized / 255.0
        # Stack into 3 channels if needed for pre-trained models
        return np.stack([normalized]*3, axis=-1)
