import librosa
import numpy as np

class Evaluator:
    def __init__(self):
        pass

    def check_pitch_accuracy(self, original_f0, generated_f0):
        # F0 RMSE
        mask = (original_f0 > 0) & (generated_f0 > 0)
        rmse = np.sqrt(np.mean((original_f0[mask] - generated_f0[mask])**2))
        return rmse

    def estimate_quality(self, audio):
        # MCD-like
        spec = np.abs(librosa.stft(audio))
        return np.mean(spec)