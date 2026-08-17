import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "stampede_model_v2.pth"

DATASET_PATH = "dataset"

IMAGE_SIZE = 224

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "critical",
    "normal",
    "warning"
]


# ============================================================
# START
# ============================================================

print("=" * 60)
print("STAMPEDE PREVENTION MODEL EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

print()
print(
    f"Total images: {len(dataset)}"
)

print(
    f"Classes: {dataset.classes}"
)


# ============================================================
# DATA LOADER
# ============================================================

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Loading ResNet18...")


model = models.resnet18(
    weights=None
)


number_of_features = (
    model.fc.in_features
)


model.fc = nn.Sequential(

    nn.Dropout(0.4),

    nn.Linear(
        number_of_features,
        3
    )
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []

all_labels = []


print()
print("Running evaluation...")


with torch.no_grad():

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(

    all_labels,

    all_predictions
)


print()
print("=" * 60)

print(
    f"ACCURACY: {accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("CLASSIFICATION REPORT")
print("=" * 60)


print(

    classification_report(

        all_labels,

        all_predictions,

        target_names=CLASSES,

        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(

    all_labels,

    all_predictions
)


print()
print("CONFUSION MATRIX")
print("=" * 60)

print()

print(
    "                 PREDICTED"
)

print(
    "              Critical Normal Warning"
)

print(
    f"ACTUAL Critical   "
    f"{matrix[0][0]:4d} "
    f"{matrix[0][1]:4d} "
    f"{matrix[0][2]:4d}"
)

print(
    f"       Normal     "
    f"{matrix[1][0]:4d} "
    f"{matrix[1][1]:4d} "
    f"{matrix[1][2]:4d}"
)

print(
    f"       Warning    "
    f"{matrix[2][0]:4d} "
    f"{matrix[2][1]:4d} "
    f"{matrix[2][2]:4d}"
)


# ============================================================
# CRITICAL RECALL
# ============================================================

critical_true = matrix[0][0]

critical_total = matrix[0].sum()

if critical_total > 0:

    critical_recall = (
        critical_true /
        critical_total
    ) * 100

else:

    critical_recall = 0


print()
print("=" * 60)

print(
    f"CRITICAL RECALL: "
    f"{critical_recall:.2f}%"
)

print("=" * 60)


# ============================================================
# COMPLETE
# ============================================================

print()
print("Evaluation complete.")
