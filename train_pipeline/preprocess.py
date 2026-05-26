import os
import subprocess
import sys

def preprocess_dataset(config):
    print("\n[PREPROCESS] Starting native DDSP-SVC preprocessing...")
    
    ddsp_dir = os.path.abspath("../DDSP")
    config_yaml = os.path.abspath(config["paths"]["ddsp_yaml"])
    python_exe = sys.executable 
    
    # Команда для виклику оригінального препроцесингу
    cmd = [python_exe, "preprocess.py", "-c", config_yaml]
    
    try:
        # Запускаємо процес у папці DDSP
        # Важливо: аудіо має лежати в папці, вказаній у train_path в config.yaml
        subprocess.run(cmd, cwd=ddsp_dir, check=True)
        print("[PREPROCESS] Preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Native preprocessing failed with error: {e}")
        sys.exit(1)