import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 1️⃣ REPRODUCIBILITY
# =========================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2️⃣ PATHS
# =========================

GENERIC_TRAIN = "dataset_generic_aug/train"
PCG_TRAIN = "dataset_pcg_aug/train"
VAL_PATH = "dataset_split/val"
TEST_NORMAL = "dataset_split/test"
TEST_SHIFT = "dataset_shifted_mild"

# =========================
# 3️⃣ TRANSFORMS
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# 4️⃣ LOADERS FUNCTION
# =========================

def get_loaders(train_path):

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform)
    test_normal_dataset = datasets.ImageFolder(TEST_NORMAL, transform=transform)
    test_shift_dataset = datasets.ImageFolder(TEST_SHIFT, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_normal_loader = DataLoader(test_normal_dataset, batch_size=8, shuffle=False)
    test_shift_loader = DataLoader(test_shift_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader, test_normal_loader, test_shift_loader

# =========================
# 5️⃣ MODEL DEFINITION
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # FIXED: 3 input channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),  # works for 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # 4 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# =========================
# 6️⃣ TRAIN FUNCTION
# =========================

def train_model(train_loader):

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}%")

    return model

# =========================
# 7️⃣ EVALUATION FUNCTION
# =========================

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# =========================
# 8️⃣ RUN EXPERIMENT
# =========================

print("\nTraining Generic Model...")
g_train, _, g_test_normal, g_test_shift = get_loaders(GENERIC_TRAIN)
model_generic = train_model(g_train)

print("\nTraining PCG Model...")
p_train, _, p_test_normal, p_test_shift = get_loaders(PCG_TRAIN)
model_pcg = train_model(p_train)

# =========================
# 9️⃣ FINAL EVALUATION
# =========================

g_normal = evaluate(model_generic, g_test_normal)
g_shift = evaluate(model_generic, g_test_shift)

p_normal = evaluate(model_pcg, p_test_normal)
p_shift = evaluate(model_pcg, p_test_shift)

print("\n===== FINAL RESULTS =====")
print(f"Generic - Normal: {g_normal:.2f}%")
print(f"Generic - Mild Shift: {g_shift:.2f}%")
print(f"Generic - Drop: {g_normal - g_shift:.2f}%")

print()

print(f"PCG - Normal: {p_normal:.2f}%")
print(f"PCG - Mild Shift: {p_shift:.2f}%")
print(f"PCG - Drop: {p_normal - p_shift:.2f}%")