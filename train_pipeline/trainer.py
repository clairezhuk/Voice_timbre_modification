import os
import subprocess
import sys
import time
import glob
from src.evaluator import Evaluator
from logger import log_validation_metrics 
import torch
import numpy as np
import librosa
import yaml

class Trainer:
    def __init__(self, config):
        self.config = config
        self.ddsp_dir = os.path.abspath("./DDSP")
        self.config_yaml = os.path.abspath(config["paths"]["ddsp_yaml"])
        self.checkpoint_dir = os.path.abspath(config["paths"]["checkpoint_dir"])
        self.val_dir = os.path.abspath("./data/dataset/val/audio")
        self.log_file = config["paths"].get("current_log_file")
        self.python_exe = sys.executable 

        with open(self.config_yaml, 'r', encoding='utf-8') as f:
            self.ddsp_config = yaml.safe_load(f)
        
        self.speedup = "2"
        self.method = "dpm-solver"
        self.kstep = self.ddsp_config["model"]["k_step_max"]

    def train(self):
        print(f"Starting training process in: {self.ddsp_dir}")
        train_cmd = [self.python_exe, "train_diff.py", "-c", self.config_yaml]
        
        while True:
            known_checkpoints = set(glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")))
            process = subprocess.Popen(train_cmd, cwd=self.ddsp_dir)
            print("\n[TRAIN] Training started. Waiting for new checkpoints...")
            
            try:
                while process.poll() is None:
                    time.sleep(60) # TIME BETWEEN LOGGING
                    
                    current_checkpoints = set(glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")))
                    new_checkpoints = current_checkpoints - known_checkpoints
                    
                    if new_checkpoints:
                        latest_ckpt = max(new_checkpoints, key=os.path.getctime)
                        print(f"\n[WATCHER] New checkpoint found: {os.path.basename(latest_ckpt)}")
                        print("[WATCHER] Stopping training to free VRAM...")
                        
                        process.terminate()
                        process.wait() 

                        self.run_validation(latest_ckpt)
                        break 
                        
            except KeyboardInterrupt:
                print("\n[STOP] Training stopped by user.")
                process.terminate()
                process.wait()
                return 

    def run_validation(self, checkpoint_path):
        val_files = glob.glob(os.path.join(self.val_dir, "*.wav"))
        if not val_files:
            print("[ERROR] No wav files found in validation directory!")
            return

        step_str = os.path.basename(checkpoint_path).replace("model_", "").replace(".pt", "")
        current_step = int(step_str) if step_str.isdigit() else 0
        current_k_max = str(self.ddsp_config["model"]["k_step_max"])
        
        f0_all, rms_all, wer_all = [], [], []
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ddsp_svc_path = os.path.join(project_root, "DDSP")
        if ddsp_svc_path not in sys.path:
            sys.path.insert(0, ddsp_svc_path)
        from DDSP.ddsp.vocoder import Units_Encoder, F0_Extractor
        from src.evaluator import Evaluator
        
        encoder = Units_Encoder(
            encoder=self.ddsp_config["data"]["encoder"], 
            encoder_ckpt=self.config["paths"]["hubert"], 
            device=self.config["device"]
        )
        f0_ext = F0_Extractor(
            f0_extractor=self.ddsp_config["data"]["f0_extractor"],
            sample_rate=44100, hop_size=512
        )
        evaluator = Evaluator(self.config["device"])

        for wav_path in val_files:
            output_wav = os.path.abspath("./temp_val_output.wav")
            
            infer_cmd = [
                self.python_exe, "main_diff.py", 
                "-i", wav_path, "-diff", checkpoint_path, 
                "-o", output_wav, "-k", current_k_max, "-id", "1", 
                "-speedup", "4", "-method", "dpm-solver", "-kstep", current_k_max
            ]
            
            subprocess.run(infer_cmd, cwd=self.ddsp_dir, stdout=subprocess.DEVNULL)
            
            if os.path.exists(output_wav):
                audio_ref, _ = librosa.load(wav_path, sr=44100)
                audio_gen, _ = librosa.load(output_wav, sr=44100)
                
                f0_ref = f0_ext.extract(audio_ref, uv_interp=True)
                f0_gen = f0_ext.extract(audio_gen, uv_interp=True)
                
                f0_all.append(evaluator.get_f0_pearson(f0_ref, f0_gen))
                rms_all.append(evaluator.get_rms_pearson(audio_ref, audio_gen))
                wer_all.append(evaluator.get_wer(wav_path, output_wav))
                os.remove(output_wav)

        file_names = [os.path.basename(f) for f in val_files]
        log_validation_metrics(self.log_file, current_step, file_names, f0_all, rms_all, wer_all)
        print(f"[EVAL] Step {current_step} done. Avg WER: {np.mean(wer_all):.4f}")
        
        del evaluator
        torch.cuda.empty_cache()
        print("=== VALIDATION COMPLETED. RETURNING TO TRAINING ===\n")