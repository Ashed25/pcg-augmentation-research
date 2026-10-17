# PCG-Based Augmentation Research

**Robustness Evaluation of Domain-Aware PCG Augmentation for Equation Image Classification in Smart Classrooms**

This repo studies whether procedural content generation (PCG) augmentation, designed around real classroom conditions, makes an image classifier more robust than generic augmentation does when the classroom conditions shift a little.

## Research question

Under a realistic shift in classroom conditions, does domain-aware procedural augmentation make the model more robust than generic augmentation?

## Dataset

- Whiteboard equation photos in four classes: `fraction`, `inequality`, `linear`, `quadratic`
- 60 images per class (240 total), converted from HEIC/JPG to grayscale PNG at 224×224
- `dataset_clean/` holds the processed images. The raw photos are not included.
- Split per class: 42 train / 9 validation / 9 test (seed 42)

## Augmentation strategies

Both methods produce 6 augmented images per training image (294 per class).

| Generic | PCG (2–3 randomly applied per image) |
|---|---|
| Rotation (±10°) | Lighting gradient |
| Translation | Glare simulation |
| Zoom (0.9–1.1) | Perspective warp |
| Brightness/contrast | Eraser smudge |
| Gaussian noise | Marker thickness variation |
| Mild blur | Motion blur |

## Experiment

The same CNN, hyperparameters and number of epochs (30) are used to train two models:

- **Model_G**: trained on generic augmentation
- **Model_P**: trained on PCG augmentation

Each model is evaluated on the normal test set and on a mild-shift test set (brightness, blur, perspective and lighting changes). The comparison covers accuracy and robustness drop (normal minus shifted).

## How to run

```bash
pip install torch torchvision opencv-python albumentations pillow pillow-heif tqdm numpy
```

Run the scripts in this order from the repo root:

```bash
python split_dataset.py            # dataset_clean -> dataset_split (train/val/test)
python generate_generic_aug.py     # -> dataset_generic_aug/train
python generate_pcg_aug.py         # -> dataset_pcg_aug/train
python create_mild_shift_test.py   # -> dataset_shifted_mild
python run_experiment.py           # trains both models and prints results
```

Other scripts:

- `prepare_dataset.py`: turns raw photos (`dataset_raw/`) into `dataset_clean/`. It is only needed if you have the raw images.
- `generate_mild_shift.py`: an alternative torchvision version of the mild-shift test set.
- `dddd.py`: generates the random equation sheet (PDF) used when writing the equations on the whiteboard.
- `check_gpu.py`: checks whether PyTorch can see a GPU.
