import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
image_dir = 'NEU-DET/IMAGES'
output_train = 'data/train'
output_test = 'data/test'

# All 6 classes
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Create folders
for label in classes:
    os.makedirs(os.path.join(output_train, label), exist_ok=True)
    os.makedirs(os.path.join(output_test, label), exist_ok=True)

# Sort images by prefix (e.g., crazing_001.jpg -> class = crazing)
all_images = os.listdir(image_dir)
for defect in classes:
    matched_imgs = [img for img in all_images if defect in img]
    train_imgs, test_imgs = train_test_split(matched_imgs, test_size=0.2, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_train, defect, img))
    for img in test_imgs:
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_test, defect, img))
