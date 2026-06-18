import os
import numpy as np
import librosa
from scipy import signal
from src.evaluator import Evaluator
from src.feature_extractor import FeatureExtractor

def align_arrays(ref, gen):
    if len(ref) == 0 or len(gen) == 0:
        return ref, gen
        
    correlation = signal.correlate(ref, gen, mode="full")
    lag = np.argmax(correlation) - (len(gen) - 1)
    
    if lag > 0:
        gen_aligned = np.pad(gen, (lag, 0), mode='constant')[:len(ref)]
        ref_aligned = ref
    else:
        ref_aligned = np.pad(ref, (abs(lag), 0), mode='constant')[:len(gen)]
        gen_aligned = gen
        
    min_len = min(len(ref_aligned), len(gen_aligned))
    return ref_aligned[:min_len], gen_aligned[:min_len]

def main():
    device = "cuda"
    base_dir = "voice_clone_project/data/tests_synthetic_1"
    output_file = "voice_clone_project/metric_test_synthetic_1_fix2.txt"
    
    paths = {
        "hubert": "voice_clone_project/models/contentvec768l12.pt",
        "rmvpe": "voice_clone_project/models/model_0.pt",
    }
    
    extractor = FeatureExtractor(paths["hubert"], paths["rmvpe"], device)
    evaluator = Evaluator(device)
    
    groups = ["speaking", "singing"]
    suffixes = [
        "1_pitch_shifted.wav",
        "2_noisy.wav",
        "3_choppy.wav",
        "4_muffled.wav"
    ]
    
    results = []
    
    for group in groups:
        ref_file = f"{group}_0_baseline.wav"
        ref_path = os.path.join(base_dir, ref_file)
        
        _, f0_ref, audio_ref = extractor.extract_features(ref_path)
        
        header = f"\n--- GROUP: {group} (Reference: {ref_file}) ---"
        print(header)
        results.append(header)
        
        for suffix in suffixes:
            test_file = f"{group}_{suffix}"
            test_path = os.path.join(base_dir, test_file)
            
            if not os.path.exists(test_path):
                continue
            
            _, f0_gen, audio_gen = extractor.extract_features(test_path)
            
            f0_ref_aligned, f0_gen_aligned = align_arrays(f0_ref, f0_gen)
            
            f0_pearson = evaluator.get_f0_pearson(f0_ref_aligned, f0_gen_aligned)
            rms_pearson = evaluator.get_rms_pearson(audio_ref, audio_gen) 
            wer = evaluator.get_wer(ref_path, test_path)
            
            output_str = (
                f"Test File: {test_file}\n"
                f"  F0 Pearson:  {f0_pearson:.4f}\n"
                f"  RMS Pearson: {rms_pearson:.4f}\n"
                f"  WER:         {wer:.4f}\n"
            )
            print(output_str)
            results.append(output_str)
            
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        
    print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    main()