import os
import csv
import numpy as np

def init_csv(log_path):
    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", 
                "f0_mean", "f0_median", "f0_std",
                "rms_mean", "rms_median", "rms_std",
                "wer_mean", "wer_median", "wer_std"
            ])


def log_validation_metrics(log_path, epoch, f0_scores, rms_scores, wer_scores):
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            np.mean(f0_scores), np.median(f0_scores), np.std(f0_scores),
            np.mean(rms_scores), np.median(rms_scores), np.std(rms_scores),
            np.mean(wer_scores), np.median(wer_scores), np.std(wer_scores)
        ])