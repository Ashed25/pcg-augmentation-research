import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A

# =========================
# CONFIGURATION
# =========================

INPUT_PATH = "dataset_split/train"
OUTPUT_PATH = "dataset_generic_aug/train"

AUG_PER_IMAGE = 6
IMAGE_SIZE = 224

# =========================
# GENERIC AUGMENTATION PIPELINE
# =========================

transform = A.Compose([
    A.Rotate(limit=10, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.4)
])

# =========================
# CREATE OUTPUT FOLDERS
# =========================

classes = os.listdir(INPUT_PATH)

for cls in classes:
    os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

# =========================
# AUGMENTATION PROCESS
# =========================

for cls in classes:
    print(f"Augmenting class: {cls}")

    input_folder = os.path.join(INPUT_PATH, cls)
    output_folder = os.path.join(OUTPUT_PATH, cls)

    images = os.listdir(input_folder)

    for img_name in tqdm(images):

        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Save original image
        cv2.imwrite(os.path.join(output_folder, img_name), image)

        # Generate augmented images
        for i in range(AUG_PER_IMAGE):

            augmented = transform(image=image)
            aug_image = augmented["image"]

            new_name = f"{os.path.splitext(img_name)[0]}_aug{i}.png"
            cv2.imwrite(os.path.join(output_folder, new_name), aug_image)

print("✅ Generic augmentation completed.")