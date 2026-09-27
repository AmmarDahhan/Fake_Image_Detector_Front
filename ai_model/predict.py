from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_base




IMAGE_SIZE = (350, 350)

IMAGE_DIR = Path("images")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}




device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )




checkpoint_files = list(
    Path(".").rglob("best_accuracy_model.pth")
)

if not checkpoint_files:
    raise FileNotFoundError(
        "best_accuracy_model.pth was not found."
    )

CHECKPOINT_PATH = checkpoint_files[0]

print(f"Model: {CHECKPOINT_PATH}")




checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

CLASS_NAMES = checkpoint["classes"]

print(f"Classes: {CLASS_NAMES}")




model = convnext_base(
    weights=None
)

model.classifier[2] = torch.nn.Linear(
    model.classifier[2].in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()




eval_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




if not IMAGE_DIR.exists():
    IMAGE_DIR.mkdir()

    raise FileNotFoundError(
        "Folder 'images' was created. "
        "Put images inside it and run the program again."
    )


image_paths = [
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in SUPPORTED_EXTENSIONS
]

if not image_paths:
    raise FileNotFoundError(
        "No images found inside the 'images' folder."
    )


# =========================================================
# Prediction
# =========================================================

print()
print("=" * 60)
print("Predictions")
print("=" * 60)


for image_path in image_paths:

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        input_tensor = eval_transform(
            image
        ).unsqueeze(0).to(device)

        with torch.inference_mode():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        predicted_index = predicted.item()

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            confidence.item() * 100
        )

        print(
            f"{image_path.name}"
            f" -> {predicted_class}"
            f" ({confidence:.2f}%)"
        )

    except Exception as e:

        print(
            f"{image_path.name}"
            f" -> ERROR: {e}"
        )