"""Diabetic Retinopathy stage detection - source package."""
import os

# Use a non-interactive matplotlib backend unless one is already configured (Colab/Jupyter set their own).
# Without this, Windows/Anaconda picks the Tk backend, which prints harmless but noisy
# "main thread is not in main loop" warnings when figures are created in tests or scripts.
os.environ.setdefault("MPLBACKEND", "Agg")
