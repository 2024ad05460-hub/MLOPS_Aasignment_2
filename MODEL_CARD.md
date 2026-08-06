# Model Card: Cats vs Dogs Classifier

## Intended use
Binary classification of cat and dog photographs for a pet-adoption platform. The model is an educational baseline and is not intended for animal-health, identity, breed, or safety decisions.

## Model
Default baseline: three-block TinyCNN serialized as a PyTorch `.pt` bundle. The training script also supports MobileNetV3-Small transfer learning using `--architecture mobilenet_v3_small --pretrained --freeze-backbone`.

## Inputs and outputs
Input: JPEG/PNG image converted to RGB and resized to 224x224. Output: probabilities for `cat` and `dog`, predicted label, confidence, latency, request ID, and model version.

## Data
The reproducible assignment workflow uses the Kaggle Dogs vs Cats dataset and stratifies it into 80% train, 10% validation, and 10% test. The ZIP contains a small generated dataset and a bootstrap artifact solely so every service and test can start without external credentials; final academic evidence should be generated after running the provided Kaggle download command.

## Evaluation
Report accuracy, macro precision, macro recall, macro F1, confusion matrix, and class-level metrics on the held-out test split. MLflow stores parameters, metrics, model artifact, loss curves, and confusion matrix.

## Limitations
Performance can degrade for cartoons, multiple animals, heavy occlusion, unusual lighting, non-cat/non-dog images, or domain shifts. The API currently forces a binary decision and does not implement an out-of-distribution rejection class.

## Monitoring
Prometheus tracks request volume, latency, HTTP errors, and predicted-class distribution. SQLite stores prediction metadata and optional true labels. `/monitoring/performance` reports delayed-label accuracy.
