import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import pillow_heif

# Register HEIF opener
pillow_heif.register_heif_opener()

# =========================
# CONFIGURATION
# =========================

RAW_DATASET_PATH = "dataset_raw"
CLEAN_DATASET_PATH = "dataset_clean"
IMAGE_SIZE = 224  # Resize to 224x224

# =========================
# CREATE OUTPUT FOLDERS
# =========================

classes = os.listdir(RAW_DATASET_PATH)

for cls in classes:
    os.makedirs(os.path.join(CLEAN_DATASET_PATH, cls), exist_ok=True)

# =========================
# PROCESS IMAGES
# =========================

for cls in classes:
    input_folder = os.path.join(RAW_DATASET_PATH, cls)
    output_folder = os.path.join(CLEAN_DATASET_PATH, cls)

    print(f"Processing class: {cls}")

    for img_name in tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)

        try:
            # Use PIL (supports HEIC now)
            img = Image.open(img_path)
            img = img.convert("L")  # Convert to grayscale
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

            save_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + ".png")
            img.save(save_path)

        except Exception as e:
            print(f"Skipped {img_name}: {e}")

print("✅ Step 1 Completed: All images resized and converted to grayscale.")