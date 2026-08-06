# Dataset note

The assignment target is the Kaggle **Dogs vs. Cats** binary image dataset. Kaggle requires a user-specific API token and acceptance of the competition rules; credentials are intentionally not included.

The ZIP contains `data/raw/demo/`, a deterministic generated bootstrap set (40 cats and 40 dogs), plus its 224x224 processed 80/10/10 split. It exists only to make the complete code/API/test/deployment path executable immediately. It is not presented as real-world Kaggle accuracy evidence.

For the final Kaggle run:

```powershell
python -m cats_dogs_mlops.download_data --mode competition --slug dogs-vs-cats --destination data/raw --force
Expand-Archive data\raw\train.zip -DestinationPath data\raw\train -Force
dvc add data/raw
dvc repro
```

The preprocessing stage detects class folders or `cat.*` / `dog.*` filenames, rejects unreadable images, converts to RGB, resizes to 224x224, and generates a deterministic stratified 80/10/10 split.
