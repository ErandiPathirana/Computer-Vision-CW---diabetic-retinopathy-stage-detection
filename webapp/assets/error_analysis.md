# Error analysis (test set, n=550)

* Overall the model made **114** errors (20.7%).
* Weakest class by recall: **Severe** (45%). Per-class recall: {'No DR': np.float64(0.98), 'Mild': np.float64(0.54), 'Moderate': np.float64(0.71), 'Severe': np.float64(0.45), 'Proliferative DR': np.float64(0.45)}.
* Most confused pairs (true -> predicted): [{'true': 'Moderate', 'predicted': 'Mild', 'count': 21}, {'true': 'Proliferative DR', 'predicted': 'Moderate', 'count': 14}, {'true': 'Mild', 'predicted': 'Moderate', 'count': 14}, {'true': 'Moderate', 'predicted': 'Proliferative DR', 'count': 11}, {'true': 'Severe', 'predicted': 'Proliferative DR', 'count': 9}].
* **68%** of the errors are between neighbouring grades, which is expected because DR severity is
  a continuum and the grade boundaries (and labels) are partly subjective.
* Mean confidence on correct predictions is 0.85 versus 0.62
  on wrong ones, so low confidence is a usable signal for sending an image to a human reader.
* Clinically the most worrying errors are severe/proliferative eyes predicted as none/mild: **4** case(s).

## Likely causes
* Class imbalance: minority grades have few training images, so their decision boundaries are weaker even with class weights.
* Resolution: downscaling to 224x224 can erase tiny lesions (microaneurysms, small haemorrhages).
* Label noise and blur/exposure variation in the source images.
