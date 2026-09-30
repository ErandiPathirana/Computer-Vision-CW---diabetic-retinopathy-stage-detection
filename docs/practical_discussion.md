# Practical and clinical discussion

## Use as a screening aid
Diabetic retinopathy is treatable when found early, but screening depends on trained graders. A model like this can
act as a first reader that orders images by urgency, so specialists spend their time on the cases that matter most.
It should support, never replace, the clinician: every image needs human review, and the app shows a Grad-CAM
heat-map and a low-confidence warning (top probability below 0.6) to help the reviewer decide how far to trust it.

## Deployment feasibility
The model is small (about 17 MB, roughly 4 million parameters) and predicts on an ordinary CPU in well under a
second, so it could run on a clinic PC or a local server without a GPU. Real deployment would additionally need
clinical validation on local data, regulatory approval, integration with the imaging software, audit logging and
monitoring for performance drift.

## Dataset bias and generalisation
The model was trained on one public dataset (APTOS 2019, single source). Different cameras, image quality,
ethnic groups and disease prevalence can change performance, and demographic information was not available, so
fairness could not be measured. External validation is required before any real use.

## Risk of errors
False negatives, especially Severe or Proliferative disease predicted as None or Mild, can delay sight-saving
treatment. False positives cause unnecessary referrals. Most errors in this project fall between neighbouring
grades, where even graders disagree. The error analysis quantifies both kinds of error on the test set.

## Privacy and ethics
Retinal images are health data. The app processes uploads in memory and never stores them; a real system would
need consent, de-identification, encryption and compliance with the applicable data-protection law. The app
clearly states that it is an educational prototype and not a medical device.

## Innovation and future work
Grad-CAM explanations, a confidence-based human-review flag, and a transparent comparison of backbones,
hyper-parameters and imbalance strategies go beyond a plain classifier. Future work: higher-resolution inputs
(EfficientNetB3/B4), ordinal losses that respect severity order, ensembles, uncertainty estimation, lesion
segmentation and prospective clinical evaluation.
