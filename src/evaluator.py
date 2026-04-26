import numpy as np
import librosa
from scipy.stats import pearsonr
import whisper
import jiwer
import warnings
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

# Вимикаємо зайві попередження від Whisper
warnings.filterwarnings("ignore", category=UserWarning)

class Evaluator:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading Whisper model for evaluation...")
        self.whisper_model = whisper.load_model("base", device=self.device)

    def get_f0_pearson(self, f0_ref, f0_gen):
        # Відкидаємо тишу
        ref_voiced = f0_ref[f0_ref > 0]
        gen_voiced = f0_gen[f0_gen > 0]
        
        if len(ref_voiced) < 2 or len(gen_voiced) < 2:
            return 0.0
            
        # 1. Переводимо Герци в логарифмічну шкалу (MIDI)
        # Це робить зсув по висоті лінійним (просто додавання константи)
        ref_midi = librosa.hz_to_midi(ref_voiced)
        gen_midi = librosa.hz_to_midi(gen_voiced)
        
        # 2. Нормалізація (Zero-mean)
        # Робимо DTW "сліпим" до загальної висоти голосу (вирішує проблему pitch_shift)
        ref_norm = ref_midi - np.mean(ref_midi)
        gen_norm = gen_midi - np.mean(gen_midi)
        
        # 3. DTW тепер порівнює тільки ФОРМУ мелодії
        distance, path = fastdtw(ref_norm.reshape(-1, 1), gen_norm.reshape(-1, 1), dist=euclidean)
        
        aligned_ref = np.array([ref_norm[i] for i, j in path])
        aligned_gen = np.array([gen_norm[j] for i, j in path])
        
        corr, _ = pearsonr(aligned_ref, aligned_gen)
        return 0.0 if np.isnan(corr) else corr

    def get_rms_pearson(self, audio_ref, audio_gen, sr=44100):
        """2. Метрика Динаміки (Гучність/Експресія)"""
        rms_ref = librosa.feature.rms(y=audio_ref)[0]
        rms_gen = librosa.feature.rms(y=audio_gen)[0]
        
        min_len = min(len(rms_ref), len(rms_gen))
        
        if min_len < 2:
            return 0.0
            
        corr, _ = pearsonr(rms_ref[:min_len], rms_gen[:min_len])
        return 0.0 if np.isnan(corr) else corr

    def get_wer(self, ref_audio_path, gen_audio_path, language="en"):
        """3. Метрика Фонем (Word Error Rate) - Bypass ffmpeg issue"""
        import librosa
        import jiwer
        
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000)
        gen_audio, _ = librosa.load(gen_audio_path, sr=16000)
        
        ref_result = self.whisper_model.transcribe(ref_audio, language=language)
        ref_text = ref_result["text"].strip().lower()
        
        gen_result = self.whisper_model.transcribe(gen_audio, language=language)
        gen_text = gen_result["text"].strip().lower()
        
        if not ref_text:
            return 1.0 if gen_text else 0.0
            
        error_rate = jiwer.wer(ref_text, gen_text)
        return error_rate