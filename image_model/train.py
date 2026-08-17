import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"

BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.00001
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 55)
print("STAMPEDE PREVENTION IMAGE MODEL - VERSION 2")
print("=" * 55)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

full_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

print(f"Total images: {len(full_dataset)}")
print(f"Classes: {full_dataset.classes}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, validation_size],
    generator=generator
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# ============================================================
# FREEZE EVERYTHING FIRST
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# UNFREEZE LAST RESNET BLOCK
# ============================================================

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# ============================================================
# NEW CLASSIFIER
# ============================================================

number_of_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.4),

    nn.Linear(
        number_of_features,
        3
    )
)


# Make sure classifier is trainable
for parameter in model.fc.parameters():
    parameter.requires_grad = True


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    filter(
        lambda parameter:
        parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# TRAINING
# ============================================================

best_validation_accuracy = 0.0


for epoch in range(NUM_EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = running_loss / total

    train_accuracy = (
        100.0 * correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss_total = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss_total += (
                loss.item() *
                images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            validation_total += (
                labels.size(0)
            )

            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_loss = (
        validation_loss_total /
        validation_total
    )

    validation_accuracy = (
        100.0 *
        validation_correct /
        validation_total
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    full_dataset.classes,

                "validation_accuracy":
                    validation_accuracy
            },
            "stampede_model_v2.pth"
        )

        print(">>> BEST MODEL SAVED")


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

print()
print("Saved model:")
print("stampede_model_v2.pth")
