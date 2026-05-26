import os
import subprocess
import sys
import time
import glob
from src.evaluator import Evaluator
from main_train import log_validation_metrics # Імпортуємо функцію запису з твого main

class Trainer:
    def __init__(self, config):
        self.config = config
        self.ddsp_dir = os.path.abspath("../DDSP")
        self.config_yaml = os.path.abspath(config["paths"]["ddsp_yaml"])
        self.checkpoint_dir = os.path.abspath(config["paths"]["checkpoint_dir"])
        self.val_dir = os.path.abspath("../data/dataset/val")
        self.log_file = config["paths"].get("current_log_file")
        self.python_exe = sys.executable 
        
        self.speedup = "10"
        self.method = "dpm-solver"
        self.kstep = "100"

    def train(self):
        print(f"Starting training process in: {self.ddsp_dir}")
        train_cmd = [self.python_exe, "train_diff.py", "-c", self.config_yaml]
        
        while True: # Глобальний цикл перезапуску навчання
            # 1. Запам'ятовуємо, які чекпоінти вже є
            known_checkpoints = set(glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")))
            
            # 2. Запускаємо навчання
            process = subprocess.Popen(train_cmd, cwd=self.ddsp_dir)
            print("\n[TRAIN] Training started. Waiting for new checkpoints...")
            
            try:
                # 3. Цикл наглядача (поки процес живий)
                while process.poll() is None:
                    time.sleep(10) # Перевіряємо кожні 10 секунд
                    
                    current_checkpoints = set(glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")))
                    new_checkpoints = current_checkpoints - known_checkpoints
                    
                    if new_checkpoints:
                        latest_ckpt = max(new_checkpoints, key=os.path.getctime)
                        print(f"\n[WATCHER] New checkpoint found: {os.path.basename(latest_ckpt)}")
                        print("[WATCHER] Stopping training to free VRAM...")
                        
                        process.terminate()
                        process.wait() # Чекаємо, поки VRAM повністю очиститься
                        
                        # 4. Запускаємо валідацію
                        self.run_validation(latest_ckpt)
                        break # Виходимо з циклу наглядача, щоб перезапустити навчання
                        
            except KeyboardInterrupt:
                print("\n[STOP] Training stopped by user.")
                process.terminate()
                process.wait()
                return # Повний вихід

    def run_validation(self, checkpoint_path):
        print(f"\n=== VALIDATION FOR CHECKPOINT: {os.path.basename(checkpoint_path)} ===")
        
        step_str = os.path.basename(checkpoint_path).replace("model_", "").replace(".pt", "")
        current_step = int(step_str) if step_str.isdigit() else 0
        
        input_wav = os.path.join(self.val_dir, "test1.wav")
        output_wav = os.path.abspath("../data/dataset/output_val.wav")
        
        infer_cmd = [
            self.python_exe, "main_diff.py", 
            "-i", input_wav, 
            "-diff", checkpoint_path, 
            "-o", output_wav, 
            "-k", "0", 
            "-id", "1", 
            "-speedup", self.speedup, 
            "-method", self.method, 
            "-kstep", self.kstep
        ]
        
        print(f"[EVAL] Generating audio for {os.path.basename(input_wav)}...")
        subprocess.run(infer_cmd, cwd=self.ddsp_dir)
        
        if os.path.exists(output_wav):
            from src.evaluator import Evaluator
            from src.feature_extractor import FeatureExtractor
            import torch
            
            print("[EVAL] Extracting features and calculating metrics...")
            
            extractor = FeatureExtractor(
                self.config["paths"]["hubert"], 
                self.config["paths"]["rmvpe"], 
                self.config["device"]
            )
            evaluator = Evaluator(self.config["device"])
            
            _, f0_ref, audio_ref = extractor.extract_features(input_wav)
            _, f0_gen, audio_gen = extractor.extract_features(output_wav)
            
            f0_score = evaluator.get_f0_pearson(f0_ref, f0_gen)
            rms_score = evaluator.get_rms_pearson(audio_ref, audio_gen)
            wer_score = evaluator.get_wer(input_wav, output_wav)
            
            f0_scores = [f0_score]
            rms_scores = [rms_score]
            wer_scores = [wer_score]
            
            print(f"[EVAL] Scores - F0: {f0_score:.4f}, RMS: {rms_score:.4f}, WER: {wer_score:.4f}")
            print("[EVAL] Saving metrics to log...")
            log_validation_metrics(self.log_file, current_step, f0_scores, rms_scores, wer_scores)
            
            del extractor
            del evaluator
            torch.cuda.empty_cache()
            
        print("=== VALIDATION COMPLETED. RETURNING TO TRAINING ===\n")