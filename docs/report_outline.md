# Report outline (one PDF, max 20 pages)

| Section | Pages | Content | Figures / screenshots to include |
|---|---|---|---|
| 1. Introduction and problem | 1 | What DR is, why early detection matters, aim, objectives | - |
| 2. Dataset (rubric 1) | 2 | APTOS 2019 source and justification, classes, distribution, stratified 70/15/15 split, ethics, limitations | class distribution, sample images, split chart |
| 3. Preprocessing (rubric 2) | 2 | crop, resize, denoise, CLAHE, unsharp mask, normalisation choice, reproducibility | steps figure, contrast/sharpness chart and numbers |
| 4. Augmentation and imbalance (rubric 3) | 1.5 | augmentations and why, class weights vs oversampling, comparison table | augmentation figure, imbalance figure, comparison table |
| 5. Model and transfer learning (rubric 4) | 2.5 | why EfficientNetB0, two-phase transfer learning, backbone comparison, hyper-parameter search | model summary (trainable/frozen), two tables |
| 6. Training strategy (rubric 5) | 1.5 | validation, callbacks, early stopping, LR schedule, overfitting control, reproducibility (seeds) | training curves |
| 7. Evaluation and error analysis (rubric 6) | 3 | accuracy, precision, recall, F1, QWK, confusion matrix, ROC, errors, Grad-CAM | metrics table, confusion matrix, ROC, misclassified gallery, Grad-CAM |
| 8. Prototype application and video (rubric 8) | 1.5 | OcuGrade features, architecture, how it was tested, hosted video URL | app screenshots (Analyze, Dataset, Training, Evaluation), video link |
| 9. Discussion, impact and ethics (rubric 9) | 2 | healthcare impact, deployment, bias, risk, privacy, limitations, future work | - |
| 10. Conclusion and references | 1 | summary, references (APTOS/Kaggle, EfficientNet, Grad-CAM, CLAHE) | - |
| Appendix (optional) | - | code structure, repository link | repo screenshot |

Tips: every figure needs a caption and a sentence saying what it shows; quote the actual numbers from
`models/metrics.json`; keep to the page limit; put the unlisted YouTube link in section 8 and on the title page.
