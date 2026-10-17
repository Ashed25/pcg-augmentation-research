import os
import torch
from torchvision import transforms
from PIL import Image
import random

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SOURCE_DIR = "dataset_split/test"
TARGET_DIR = "dataset_shifted_mild"

# Mild but real-world shift
shift_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3),
])

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for class_name in os.listdir(SOURCE_DIR):
    class_source = os.path.join(SOURCE_DIR, class_name)
    class_target = os.path.join(TARGET_DIR, class_name)

    ensure_dir(class_target)

    for img_name in os.listdir(class_source):
        img_path = os.path.join(class_source, img_name)
        save_path = os.path.join(class_target, img_name)

        image = Image.open(img_path).convert("RGB")
        shifted_image = shift_transform(image)
        shifted_image.save(save_path)

print("Mild shifted dataset generated successfully.")