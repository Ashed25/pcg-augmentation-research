import os
import cv2
import random
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================

INPUT_PATH = "dataset_split/test"
OUTPUT_PATH = "dataset_shifted_mild"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# MILD SHIFT TRANSFORMS
# =========================

def slight_brightness(image):
    factor = random.uniform(0.9, 1.1)
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)

def slight_blur(image):
    return cv2.GaussianBlur(image, (3,3), 0)

def slight_perspective(image):
    h, w = image.shape
    shift = 5

    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [random.randint(0,shift), random.randint(0,shift)],
        [w-random.randint(0,shift), random.randint(0,shift)],
        [random.randint(0,shift), h-random.randint(0,shift)],
        [w-random.randint(0,shift), h-random.randint(0,shift)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w,h))

def mild_lighting_gradient(image):
    h, w = image.shape
    gradient = np.tile(np.linspace(1.05, 0.95, h), (w, 1)).T
    img = image.astype(np.float32) * gradient
    return np.clip(img, 0, 255).astype(np.uint8)

# =========================
# CREATE FOLDERS
# =========================

classes = os.listdir(INPUT_PATH)

for cls in classes:
    os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

# =========================
# GENERATE SHIFTED TEST
# =========================

for cls in classes:
    print(f"Creating mild shift for class: {cls}")

    input_folder = os.path.join(INPUT_PATH, cls)
    output_folder = os.path.join(OUTPUT_PATH, cls)

    images = os.listdir(input_folder)

    for img_name in tqdm(images):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        aug = image.copy()

        transforms = [
            slight_brightness,
            slight_blur,
            slight_perspective,
            mild_lighting_gradient
        ]

        selected = random.sample(transforms, k=random.randint(1,2))

        for t in selected:
            aug = t(aug)

        cv2.imwrite(os.path.join(output_folder, img_name), aug)

print("✅ Mild shifted test set created.")