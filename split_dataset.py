import os
import shutil
import random

# =========================
# CONFIGURATION
# =========================

CLEAN_DATASET_PATH = "dataset_clean"
OUTPUT_PATH = "dataset_split"

TRAIN_COUNT = 42
VAL_COUNT = 9
TEST_COUNT = 9

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# =========================
# CREATE OUTPUT FOLDERS
# =========================

splits = ["train", "val", "test"]
classes = os.listdir(CLEAN_DATASET_PATH)

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

# =========================
# SPLIT PER CLASS
# =========================

for cls in classes:
    print(f"Splitting class: {cls}")

    class_path = os.path.join(CLEAN_DATASET_PATH, cls)
    images = os.listdir(class_path)

    if len(images) != 60:
        print(f"Warning: {cls} has {len(images)} images instead of 60")

    random.shuffle(images)

    train_imgs = images[:TRAIN_COUNT]
    val_imgs = images[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_imgs = images[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    # Copy files
    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_PATH, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_PATH, "val", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_PATH, "test", cls, img)
        )

print("✅ Exact 42 / 9 / 9 split completed.")