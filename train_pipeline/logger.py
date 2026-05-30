import os
import time
import numpy as np

def init_txt(log_path, model_size="Unknown"):
    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, mode='w', encoding='utf-8') as f:
            f.write(f"=== TRAINING LOG ===\n")
            f.write(f"Model Size: {model_size}\n")
            f.write("-" * 50 + "\n")

def log_validation_metrics(log_path, step, file_names, f0_scores, rms_scores, wer_scores):
    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write(f"\n[VALIDATION AT STEP {step}]\n")
        f.write(f"{'File Name':<30} | {'F0 Pear.':<10} | {'RMS Pear.':<10} | {'WER':<10}\n")
        f.write("-" * 65 + "\n")
        
        for i in range(len(file_names)):
            f.write(f"{file_names[i]:<30} | {f0_scores[i]:<10.4f} | {rms_scores[i]:<10.4f} | {wer_scores[i]:<10.4f}\n")
        
        f.write("-" * 65 + "\n")
        f.write(f"{'AVERAGE':<30} | {np.mean(f0_scores):<10.4f} | {np.mean(rms_scores):<10.4f} | {np.mean(wer_scores):<10.4f}\n")
        f.write("=" * 65 + "\n")