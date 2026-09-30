# Diabetic Retinopathy Stage Detection (Computer Vision coursework)

Classifies retinal fundus photographs into five diabetic retinopathy stages using transfer learning
(EfficientNetB0), with CLAHE-based preprocessing, augmentation, class-imbalance handling, evaluation,
error analysis and Grad-CAM explanations. Educational prototype, **not a medical device**.

## Pipeline
```
Drive zip (APTOS 2019) -> validation & EDA -> stratified 70/15/15 split -> preprocessing
(crop, resize, denoise, CLAHE, edge enhancement) -> augmentation + class weights -> EfficientNetB0
transfer learning (2 phases) -> tuning experiments -> evaluation (accuracy, precision, recall, F1,
confusion matrix, ROC, QWK) -> error analysis + Grad-CAM -> export -> web app
```

## Repository layout
```
src/            config, data, preprocess, augment, model, train, experiments, evaluate, gradcam, export_assets
notebooks/      DR_Stage_Detection.ipynb  (Colab runner, imports from src/)
tests/          smoke and integration tests
app/            Gradio fallback demo
webapp/         OcuGrade web app: server.py, inference.py, report.py, static/ (HTML, CSS, JS), assets/
docs/           assignment brief + checklist, model card, progress notes
data/README.md  how to provide the dataset (images are not in the repo)
```

## How to run in Google Colab
1. Put `Images and train file.zip` (APTOS images + labelled CSV) in your Google Drive (see `data/README.md`).
2. Open the notebook: `https://colab.research.google.com/github/ErandiPathirana/Computer-Vision-CW---diabetic-retinopathy-stage-detection/blob/main/notebooks/DR_Stage_Detection.ipynb`
3. Runtime > Change runtime type > **T4 GPU**, then run all cells from top to bottom.
4. The last cell exports `export.zip` (model + figures) for the web app.

## How to run locally (tests and web app)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```
Use the same TensorFlow version as the Colab run (printed by the notebook).

## Web app (OcuGrade)
After the Colab run, unzip `export.zip` into the project folder (it creates `models/` and `webapp/assets/`), then:
```
.venv\Scripts\activate
python -m webapp.server --open        # or double-click run_app.bat
```
Open http://localhost:8080. Pages: **Analyze** (upload, pipeline viewer, severity gauge, probabilities, Grad-CAM with opacity
slider, low-confidence warning, PDF report), **Dataset & Pipeline**, **Model & Training**, **Evaluation** (metrics, confusion
matrix, ROC, error analysis) and **Impact & Ethics**. Uploaded images are processed in memory and never stored.
Optional demo images for the "Try a sample" buttons go in `webapp/samples/` (git-ignored, competition data).

## Limitations and ethics
Small single-source dataset, class imbalance, 224x224 downscaling, noisy labels and unknown demographics.
False negatives on severe disease are the main clinical risk; a clinician must review every result.
