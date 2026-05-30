import os
import csv
import numpy as np
from preprocess import preprocess_dataset
from trainer import Trainer
from logger import log_validation_metrics, init_csv

CONFIG = {
    "device": "cuda",
    "paths": {
        "train_dir": "./data/dataset/train",
        "val_dir": "./data/dataset/val",
        "checkpoint_dir": "./my_model",
        "ddsp_yaml": "./DDSP/config.yaml",
        "log_dir": "./training_log"
    },
    "training": {
        "log_index": 1
    }
}

def main():
    os.makedirs(CONFIG["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["paths"]["log_dir"], exist_ok=True)

    log_file = os.path.join(CONFIG["paths"]["log_dir"], f'training_log_{CONFIG["training"]["log_index"]}.csv')
    init_csv(log_file)
    CONFIG["paths"]["current_log_file"] = log_file

    #print("=== STEP 1: Preprocessing ===")
    #preprocess_dataset(CONFIG)

    print("\n=== STEP 2: Training ===")
    trainer = Trainer(CONFIG)
    
    trainer.train()

if __name__ == "__main__":
    main()