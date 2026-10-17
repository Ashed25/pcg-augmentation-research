import os
import cv2
import random
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

INPUT_PATH = "dataset_split/train"
OUTPUT_PATH = "dataset_pcg_aug/train"

AUG_PER_IMAGE = 6
IMAGE_SIZE = 224
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# PCG TRANSFORM FUNCTIONS
# =========================

def apply_lighting_gradient(image):
    h, w = image.shape
    gradient = np.tile(np.linspace(1.2, 0.8, h), (w, 1)).T
    image = image.astype(np.float32) * gradient
    return np.clip(image, 0, 255).astype(np.uint8)

def apply_glare(image):
    h, w = image.shape
    overlay = image.copy().astype(np.float32)

    center_x = random.randint(w//4, 3*w//4)
    center_y = random.randint(h//4, 3*h//4)
    radius = random.randint(20, 50)

    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < radius:
                overlay[y, x] += (1 - dist/radius) * 80

    return np.clip(overlay, 0, 255).astype(np.uint8)

def apply_perspective(image):
    h, w = image.shape

    shift = 10
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [random.randint(0,shift), random.randint(0,shift)],
        [w-random.randint(0,shift), random.randint(0,shift)],
        [random.randint(0,shift), h-random.randint(0,shift)],
        [w-random.randint(0,shift), h-random.randint(0,shift)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w,h))

def apply_smudge(image):
    h, w = image.shape
    overlay = image.copy().astype(np.float32)

    x1 = random.randint(0, w-50)
    y1 = random.randint(0, h-10)
    x2 = x1 + random.randint(30, 80)
    y2 = y1 + random.randint(5, 15)

    overlay[y1:y2, x1:x2] += 40
    return np.clip(overlay, 0, 255).astype(np.uint8)

def apply_marker_variation(image):
    kernel_size = random.choice([2,3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if random.random() > 0.5:
        return cv2.dilate(image, kernel, iterations=1)
    else:
        return cv2.erode(image, kernel, iterations=1)

def apply_motion_blur(image):
    kernel_size = random.choice([3,5])
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

# =========================
# CREATE OUTPUT FOLDERS
# =========================

classes = os.listdir(INPUT_PATH)

for cls in classes:
    os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

# =========================
# PCG AUGMENTATION PROCESS
# =========================

for cls in classes:
    print(f"PCG Augmenting class: {cls}")

    input_folder = os.path.join(INPUT_PATH, cls)
    output_folder = os.path.join(OUTPUT_PATH, cls)

    images = os.listdir(input_folder)

    for img_name in tqdm(images):

        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Save original
        cv2.imwrite(os.path.join(output_folder, img_name), image)

        # Generate PCG augmentations
        for i in range(AUG_PER_IMAGE):

            aug = image.copy()

            transforms = [
                apply_lighting_gradient,
                apply_glare,
                apply_perspective,
                apply_smudge,
                apply_marker_variation,
                apply_motion_blur
            ]

            selected = random.sample(transforms, k=random.randint(2,3))

            for t in selected:
                aug = t(aug)

            new_name = f"{os.path.splitext(img_name)[0]}_pcg{i}.png"
            cv2.imwrite(os.path.join(output_folder, new_name), aug)

print("✅ PCG augmentation completed.")