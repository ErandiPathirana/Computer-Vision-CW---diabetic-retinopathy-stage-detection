# Coursework brief and requirement checklist

**Module:** Computer Vision (BSc (Hons) Computer Science, NIBM) - Individual - Project report + video demonstration.
**Task:** Diabetic Retinopathy stage detection using Kaggle retinal images.

Deliverables: commented codebase, ONE PDF report (max 20 pages) with graphical evidence for every step,
and a recorded prototype demonstration whose hosted video URL is included in the report.
All work must be original (plagiarism = disqualification).

| Rubric row (marks) | What is required | Where it is implemented | Evidence to screenshot |
|---|---|---|---|
| 1. Problem & dataset (10) | DR significance, classes, distribution, splits, ethics, limitations | `src/data.py`, `results/eda/eda_summary.md` | class distribution, sample images, split chart |
| 2. Preprocessing (10) | contrast, resize, normalisation, noise removal, edge enhancement, reproducible | `src/preprocess.py` | `preprocessing_steps.png`, `preprocessing_metrics.png` |
| 3. Augmentation & balancing (10) | justified augmentation, class imbalance handled | `src/augment.py`, `src/experiments.py` | augmentation figure, imbalance figure and comparison table |
| 4. CNN & transfer learning (20) | suitable backbone, correct transfer learning, tuning | `src/model.py`, `src/experiments.py` | model summary (trainable/frozen), backbone + hyper-parameter tables |
| 5. Training strategy (10) | validation, callbacks, early stopping, LR schedule, overfitting control | `src/train.py` | training curves (phase 1, phase 2, combined) |
| 6. Evaluation (15) | accuracy, precision, recall, F1, confusion matrix, curves, error analysis | `src/evaluate.py` | metrics table, confusion matrix, ROC, misclassified gallery, Grad-CAM |
| 7. Code quality (10) | modular, commented, reproducible | whole repo, `tests/`, `README.md` | repo screenshot, passing tests |
| 8. Report & video (10) | professional report, hosted video link | `docs/report_outline.md` (to write) | screen recording of the web app |
| 9. Innovation & impact (5) | real-world impact, feasibility, limits, ethics, future work | `docs/practical_discussion.md`, Grad-CAM, low-confidence flag | web app + discussion section |
