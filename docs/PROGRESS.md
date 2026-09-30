# Project progress notes

## Finished (code written, needs a Colab run)
* `src/config.py` paths (Colab + Windows), constants (IMG_SIZE 224, BATCH 32, SEED 42, LR 1e-3 / 1e-5, dropout 0.3, 30 unfrozen layers, 70/15/15 split)
* `src/data.py` load from Google Drive zip, validation, EDA, stratified split
* `src/preprocess.py` crop, pad/resize, denoise, CLAHE, unsharp mask, cache, figures
* `src/augment.py` tf.data pipeline, augmentation, class weights, oversampling
* `src/model.py`, `src/train.py`, `src/experiments.py` (backbone comparison, hyper-parameter search, imbalance comparison)
* `src/evaluate.py`, `src/gradcam.py`, `src/export_assets.py`
* `notebooks/DR_Stage_Detection.ipynb` Colab runner, `tests/`, `Dockerfile`, docs

## Remaining
1. Run the notebook in Colab (T4 GPU), download `export.zip`, unzip into the project folder (creates `models/` and `webapp/assets/`).
2. Web app is written (webapp/); test it with the real model, add sample images and README screenshots.
3. Report (max 20 pages) + hosted demo video.

## Commands (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
python -m webapp.server --open   # http://localhost:8080
```
Model files: `results/best_model.keras` (training output) is copied to `models/best_model.keras` by `src/export_assets.py`.
Install the same TensorFlow version locally as the one printed by the Colab notebook.
