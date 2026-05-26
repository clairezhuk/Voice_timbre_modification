import os
import csv
import numpy as np
from preprocess import preprocess_dataset
from trainer import Trainer

CONFIG = {
    "device": "cuda",
    "paths": {
        "train_dir": "../data/dataset/train",
        "val_dir": "../data/dataset/val",
        "checkpoint_dir": "../my_model",
        "ddsp_yaml": "../DDSP/config.yaml",
        "log_dir": "../training_log"
    },
    "training": {
        "log_index": 1
    }
}

def init_csv(log_path):
    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", 
                "f0_mean", "f0_median", "f0_std", "f0_min", "f0_max",
                "rms_mean", "rms_median", "rms_std", "rms_min", "rms_max",
                "wer_mean", "wer_median", "wer_std", "wer_min", "wer_max"
            ])

def log_validation_metrics(log_path, epoch, f0_scores, rms_scores, wer_scores):
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            np.mean(f0_scores), np.median(f0_scores), np.std(f0_scores), np.min(f0_scores), np.max(f0_scores),
            np.mean(rms_scores), np.median(rms_scores), np.std(rms_scores), np.min(rms_scores), np.max(rms_scores),
            np.mean(wer_scores), np.median(wer_scores), np.std(wer_scores), np.min(wer_scores), np.max(wer_scores)
        ])

def main():
    os.makedirs(CONFIG["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["paths"]["log_dir"], exist_ok=True)

    log_file = os.path.join(CONFIG["paths"]["log_dir"], f'training_log_{CONFIG["training"]["log_index"]}.csv')
    init_csv(log_file)
    CONFIG["paths"]["current_log_file"] = log_file

    print("=== STEP 1: Preprocessing ===")
    preprocess_dataset(CONFIG)

    print("\n=== STEP 2: Training ===")
    trainer = Trainer(CONFIG)
    
    trainer.train()

if __name__ == "__main__":
    main()