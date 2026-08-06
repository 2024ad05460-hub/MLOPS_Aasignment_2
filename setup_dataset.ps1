$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================="
Write-Host " Cats vs Dogs Dataset Setup"
Write-Host "==============================================="
Write-Host ""

# ------------------------------------------------------------
# 1. Confirm that the script is running from the project root
# ------------------------------------------------------------

if (-not (Test-Path ".\data")) {
    Write-Host "ERROR: The data folder was not found."
    Write-Host "Run this script from the project root:"
    Write-Host "C:\Users\ASUS\ANN_Projects\Cats_Dogs_MLOps_Assignment2"
    exit 1
}

# ------------------------------------------------------------
# 2. Activate virtual environment
# ------------------------------------------------------------

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}
else {
    Write-Host "ERROR: Virtual environment .venv was not found."
    exit 1
}

Write-Host "Python environment:"
python --version

# ------------------------------------------------------------
# 3. Install required packages
# ------------------------------------------------------------

Write-Host ""
Write-Host "Installing required packages..."

python -m pip install --upgrade pip
python -m pip install kaggle pillow tqdm scikit-learn

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Package installation failed."
    exit $LASTEXITCODE
}

# ------------------------------------------------------------
# 4. Define paths
# ------------------------------------------------------------

$RawDirectory = "data\raw"
$ZipPath = "data\raw\dog-and-cat-classification-dataset.zip"
$ExtractPath = "data\raw\dog-and-cat-classification-dataset"
$ProcessedPath = "data\processed"

New-Item -ItemType Directory -Force $RawDirectory | Out-Null

# ------------------------------------------------------------
# 5. Download dataset only if the ZIP is missing
# ------------------------------------------------------------

if (-not (Test-Path $ZipPath)) {
    Write-Host ""
    Write-Host "Dataset ZIP not found."
    Write-Host "Downloading from Kaggle..."

    python -m kaggle datasets download `
        -d bhavikjikadara/dog-and-cat-classification-dataset `
        -p $RawDirectory

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Kaggle download failed."
        Write-Host "Check your Kaggle API token:"
        Write-Host "$env:USERPROFILE\.kaggle\kaggle.json"
        exit $LASTEXITCODE
    }
}
else {
    Write-Host ""
    Write-Host "Dataset ZIP already exists. Download skipped."
}

# ------------------------------------------------------------
# 6. Extract dataset
# ------------------------------------------------------------

if (-not (Test-Path $ExtractPath)) {
    Write-Host ""
    Write-Host "Extracting dataset..."

    Expand-Archive `
        -Path $ZipPath `
        -DestinationPath $ExtractPath `
        -Force
}
else {
    Write-Host ""
    Write-Host "Extracted dataset already exists. Extraction skipped."
}

# ------------------------------------------------------------
# 7. Display extracted directories
# ------------------------------------------------------------

Write-Host ""
Write-Host "Extracted directories:"
Write-Host "-----------------------------------------------"

Get-ChildItem $ExtractPath -Directory -Recurse |
    Select-Object FullName

# ------------------------------------------------------------
# 8. Count source images
# ------------------------------------------------------------

$SourceImages = Get-ChildItem $ExtractPath -File -Recurse |
    Where-Object {
        $_.Extension.ToLower() -in @(".jpg", ".jpeg", ".png")
    }

Write-Host ""
Write-Host "Source image count: $($SourceImages.Count)"

if ($SourceImages.Count -eq 0) {
    Write-Host "ERROR: No images were found after extraction."
    exit 1
}

# ------------------------------------------------------------
# 9. Embedded Python preprocessing program
# ------------------------------------------------------------

Write-Host ""
Write-Host "Validating and preparing the dataset..."
Write-Host ""

$PythonCode = @'
from __future__ import annotations

import hashlib
import json
import random
import shutil
import warnings
from pathlib import Path

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# Allow PIL to load mildly truncated images.
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings(
    "ignore",
    message="Truncated File Read"
)

SOURCE = Path(
    "data/raw/dog-and-cat-classification-dataset"
)

DESTINATION = Path(
    "data/processed"
)

IMAGE_SIZE = (224, 224)
SEED = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

random.seed(SEED)


def determine_label(path: Path) -> str | None:
    """
    Determine the class from the actual parent directory.

    Expected dataset structure:

    PetImages/
        Cat/
        Dog/
    """

    parent_names = {
        parent.name.lower()
        for parent in path.parents
    }

    if "cat" in parent_names:
        return "cat"

    if "dog" in parent_names:
        return "dog"

    # Fallback for datasets where labels are in filenames.
    filename = path.name.lower()

    if filename.startswith("cat"):
        return "cat"

    if filename.startswith("dog"):
        return "dog"

    return None


def validate_image(path: Path) -> bool:
    """
    Verify that an image can be opened and converted to RGB.
    """

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.convert("RGB")

        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError
    ):
        return False


def split_items(
    items: list[Path]
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Create a deterministic stratified split for one class:

    80% training
    10% validation
    10% testing
    """

    shuffled = items.copy()

    random.Random(SEED).shuffle(shuffled)

    total = len(shuffled)

    train_count = int(total * 0.80)
    validation_count = int(total * 0.10)

    train_end = train_count
    validation_end = train_count + validation_count

    train_items = shuffled[:train_end]

    validation_items = shuffled[
        train_end:validation_end
    ]

    test_items = shuffled[
        validation_end:
    ]

    return (
        train_items,
        validation_items,
        test_items
    )


def unique_filename(
    source: Path,
    label: str
) -> str:
    """
    Create a deterministic unique filename to prevent collisions.
    """

    digest = hashlib.sha1(
        str(source.resolve()).encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{label}_{source.stem}_{digest}.jpg"
    )


def process_image(
    source: Path,
    destination: Path
) -> None:
    """
    Convert the image to RGB and resize it to 224x224.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        image.save(
            destination,
            format="JPEG",
            quality=95,
            optimize=True
        )


# ------------------------------------------------------------
# Discover all source images
# ------------------------------------------------------------

all_images = sorted(
    path
    for path in SOURCE.rglob("*")
    if path.is_file()
    and path.suffix.lower() in VALID_EXTENSIONS
)

print(
    f"Discovered images: {len(all_images)}"
)

if not all_images:
    raise RuntimeError(
        "No source images were found."
    )

# ------------------------------------------------------------
# Validate and classify images
# ------------------------------------------------------------

classified: dict[str, list[Path]] = {
    "cat": [],
    "dog": []
}

corrupt_images: list[str] = []
unclassified_images: list[str] = []

for image_path in all_images:
    label = determine_label(image_path)

    if label is None:
        unclassified_images.append(
            str(image_path)
        )
        continue

    if not validate_image(image_path):
        corrupt_images.append(
            str(image_path)
        )
        continue

    classified[label].append(image_path)

print(
    f"Valid cat images: "
    f"{len(classified['cat'])}"
)

print(
    f"Valid dog images: "
    f"{len(classified['dog'])}"
)

print(
    f"Corrupt images: "
    f"{len(corrupt_images)}"
)

print(
    f"Unclassified images: "
    f"{len(unclassified_images)}"
)

if not classified["cat"]:
    raise RuntimeError(
        "No valid cat images were identified."
    )

if not classified["dog"]:
    raise RuntimeError(
        "No valid dog images were identified."
    )

# ------------------------------------------------------------
# Remove old processed data
# ------------------------------------------------------------

if DESTINATION.exists():
    print(
        "Removing existing processed dataset..."
    )

    shutil.rmtree(DESTINATION)

# ------------------------------------------------------------
# Split each class separately
# ------------------------------------------------------------

dataset_splits: dict[
    str,
    dict[str, list[Path]]
] = {}

for label in ("cat", "dog"):
    train_items, val_items, test_items = (
        split_items(classified[label])
    )

    dataset_splits[label] = {
        "train": train_items,
        "val": val_items,
        "test": test_items
    }

# ------------------------------------------------------------
# Process and save images
# ------------------------------------------------------------

manifest_records: list[
    dict[str, object]
] = []

summary: dict[
    str,
    dict[str, int]
] = {
    "cat": {},
    "dog": {}
}

failed_processing: list[str] = []

for label in ("cat", "dog"):
    for split_name in (
        "train",
        "val",
        "test"
    ):
        source_files = (
            dataset_splits[label][split_name]
        )

        output_directory = (
            DESTINATION /
            split_name /
            label
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        successful_count = 0

        for source_file in source_files:
            output_file = (
                output_directory /
                unique_filename(
                    source_file,
                    label
                )
            )

            try:
                process_image(
                    source_file,
                    output_file
                )

                successful_count += 1

                manifest_records.append(
                    {
                        "source": str(
                            source_file
                        ),
                        "output": str(
                            output_file
                        ),
                        "split": split_name,
                        "label": label,
                        "width": IMAGE_SIZE[0],
                        "height": IMAGE_SIZE[1],
                        "channels": 3
                    }
                )

            except Exception as error:
                failed_processing.append(
                    f"{source_file}: {error}"
                )

        summary[label][split_name] = (
            successful_count
        )

# ------------------------------------------------------------
# Save dataset manifest
# ------------------------------------------------------------

manifest = {
    "dataset": (
        "bhavikjikadara/"
        "dog-and-cat-classification-dataset"
    ),
    "seed": SEED,
    "image_size": list(IMAGE_SIZE),
    "color_mode": "RGB",
    "split_ratio": {
        "train": 0.80,
        "validation": 0.10,
        "test": 0.10
    },
    "summary": summary,
    "source_image_count": len(all_images),
    "valid_cat_count": len(
        classified["cat"]
    ),
    "valid_dog_count": len(
        classified["dog"]
    ),
    "corrupt_images": corrupt_images,
    "unclassified_images": (
        unclassified_images
    ),
    "failed_processing": (
        failed_processing
    ),
    "records": manifest_records
}

manifest_path = (
    DESTINATION /
    "dataset_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print()
print(
    "Dataset preparation completed."
)
print()

total_processed = 0

for split_name in (
    "train",
    "val",
    "test"
):
    for label in (
        "cat",
        "dog"
    ):
        folder = (
            DESTINATION /
            split_name /
            label
        )

        count = len(
            list(
                folder.glob("*.jpg")
            )
        )

        total_processed += count

        print(
            f"{split_name:5s}/"
            f"{label:3s}: {count}"
        )

print()
print(
    f"Total processed images: "
    f"{total_processed}"
)

print(
    f"Failed during processing: "
    f"{len(failed_processing)}"
)

print(
    f"Manifest saved to: "
    f"{manifest_path}"
)

if total_processed == 0:
    raise RuntimeError(
        "No images were processed."
    )
'@

# Pass embedded Python code to Python through stdin.
$PythonCode | python -

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Dataset preparation failed."
    exit $LASTEXITCODE
}

# ------------------------------------------------------------
# 10. Final PowerShell verification
# ------------------------------------------------------------

Write-Host ""
Write-Host "==============================================="
Write-Host " Final Processed Dataset"
Write-Host "==============================================="

$Splits = @(
    "train",
    "val",
    "test"
)

$Classes = @(
    "cat",
    "dog"
)

$GrandTotal = 0

foreach ($Split in $Splits) {
    foreach ($Class in $Classes) {
        $Folder = Join-Path `
            $ProcessedPath `
            "$Split\$Class"

        if (Test-Path $Folder) {
            $Count = (
                Get-ChildItem `
                    $Folder `
                    -File `
                    -Filter "*.jpg"
            ).Count
        }
        else {
            $Count = 0
        }

        $GrandTotal += $Count

        Write-Host (
            "{0}/{1} : {2}" -f `
            $Split, `
            $Class, `
            $Count
        )
    }
}

Write-Host ""
Write-Host "Total processed images: $GrandTotal"

Write-Host ""
Write-Host "Processed data location:"

if (Test-Path $ProcessedPath) {
    Write-Host (
        Resolve-Path $ProcessedPath
    )
}

Write-Host ""
Write-Host "Dataset manifest:"

$ManifestPath = Join-Path `
    $ProcessedPath `
    "dataset_manifest.json"

if (Test-Path $ManifestPath) {
    Write-Host (
        Resolve-Path $ManifestPath
    )
}

Write-Host ""
Write-Host "Dataset setup completed successfully."