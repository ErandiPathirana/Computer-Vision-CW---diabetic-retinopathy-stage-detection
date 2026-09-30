# Model card: DR stage classifier (educational prototype)

* **Task:** classify a retinal fundus photograph into 5 DR stages (0 No DR ... 4 Proliferative DR).
* **Model:** EfficientNetB0 (ImageNet weights) + GlobalAveragePooling -> BatchNorm -> Dropout(0.3) -> Dense(5, softmax); two-phase transfer learning.
* **Data:** APTOS 2019 Blindness Detection (Kaggle), stratified 70/15/15 split. See `data/README.md`.
* **Input:** RGB fundus image -> crop, resize 224x224, denoise, CLAHE, unsharp mask (`src/preprocess.py`).
* **Metrics:** see `models/metrics.json` after training (accuracy, macro precision/recall/F1, quadratic weighted kappa, confusion matrix).
* **Intended use:** learning and demonstration only. Not a medical device, not validated clinically.
* **Limitations:** small single-source dataset, class imbalance, downscaled resolution, noisy labels, unknown demographics.
* **Risks:** false negatives on severe disease could delay treatment; over-reliance on automation. A qualified clinician must review every result.
* **Explainability:** Grad-CAM heat-maps and a low-confidence warning (top probability < 0.6).
