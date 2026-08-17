import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "stampede_model_v2.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "critical",
    "normal",
    "warning"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 50)
print("STAMPEDE RISK ANALYSIS")
print("=" * 50)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Load ResNet18
model = models.resnet18(weights=None)


# Same architecture used during training
number_of_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(number_of_features, 3)
)


# Load trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GET IMAGE PATH
# ============================================================

if len(sys.argv) < 2:

    print()
    print("Usage:")
    print("python predict.py \"path_to_image.jpg\"")
    print()

    sys.exit()


image_path = sys.argv[1]


if not os.path.exists(image_path):

    print()
    print("ERROR: Image not found.")
    print(image_path)

    sys.exit()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as error:

    print("ERROR opening image:")
    print(error)

    sys.exit()


# ============================================================
# PREPROCESS
# ============================================================

image_tensor = transform(
    image
)

image_tensor = image_tensor.unsqueeze(
    0
)

image_tensor = image_tensor.to(
    DEVICE
)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]


# ============================================================
# RESULT
# ============================================================

predicted_index = torch.argmax(
    probabilities
).item()

predicted_class = CLASSES[
    predicted_index
]

confidence = (
    probabilities[predicted_index].item()
    * 100
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 50)

print(
    f"PREDICTION: "
    f"{predicted_class.upper()}"
)

print(
    f"CONFIDENCE: "
    f"{confidence:.2f}%"
)

print("=" * 50)

print()

for i, class_name in enumerate(CLASSES):

    probability = (
        probabilities[i].item()
        * 100
    )

    print(
        f"{class_name.upper():10} "
        f"{probability:.2f}%"
    )

print()
print("=" * 50)
