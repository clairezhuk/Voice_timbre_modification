import librosa
import numpy as np

class Evaluator:
    def __init__(self):
        pass

    def check_pitch_accuracy(self, original_f0, generated_f0):
        min_len = min(len(original_f0), len(generated_f0))
        orig = original_f0[:min_len]
        gen = generated_f0[:min_len]
        
        mask = (orig > 0) & (gen > 0)
        if not np.any(mask):
            return 0.0
            
        rmse = np.sqrt(np.mean((orig[mask] - gen[mask])**2))
        return rmse

    def estimate_quality(self, audio):
        spec = np.abs(librosa.stft(audio))
        return np.mean(spec)