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
    base_dir = "voice_clone_project/data/tests_for_metrics"
    output_file = "voice_clone_project/metric_test.txt"
    
    paths = {
        "hubert": "voice_clone_project/models/contentvec768l12.pt",
        "rmvpe": "voice_clone_project/models/model_0.pt",
    }
    
    extractor = FeatureExtractor(paths["hubert"], paths["rmvpe"], device)
    evaluator = Evaluator(device)
    
    groups = {
        "I_cry": ["I_cry_0.wav", "I_cry_1_melody.wav", "I_cry_2_melody_rythm.wav", "I_cry_3_noise.wav"],
        "papperony": ["papperony_0.wav", "papperony_1_close.wav", "papperony_2_f.wav", "papperony_3_text.wav"],
        "please_dont": ["please_dont_0.wav", "please_dont_1_intonation.wav", "please_dont_2_mumling.wav", "please_dont_3_creap.wav"]
    }
    
    results = []
    
    for group_name, files in groups.items():
        ref_file = files[0]
        ref_path = os.path.join(base_dir, ref_file)
        
        _, f0_ref, audio_ref = extractor.extract_features(ref_path)
        
        group_header = f"\n--- GROUP: {group_name} (Reference: {ref_file}) ---"
        print(group_header)
        results.append(group_header)
        
        for test_file in files[1:]:
            test_path = os.path.join(base_dir, test_file)
            
            _, f0_gen, audio_gen = extractor.extract_features(test_path)
            
            f0_ref_aligned, f0_gen_aligned = align_arrays(f0_ref, f0_gen)
            
            rms_ref = librosa.feature.rms(y=audio_ref)[0]
            rms_gen = librosa.feature.rms(y=audio_gen)[0]
            rms_ref_aligned, rms_gen_aligned = align_arrays(rms_ref, rms_gen)
            
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