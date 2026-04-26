import numpy as np
import librosa
from scipy.stats import pearsonr
import whisper
import jiwer
import warnings

# Вимикаємо зайві попередження від Whisper
warnings.filterwarnings("ignore", category=UserWarning)

class Evaluator:
    def __init__(self, device="cuda"):
        self.device = device
        # Завантажуємо базову модель Whisper (її достатньо для перевірки фонем англійською/іншими мовами)
        # Вона "важить" близько 140 МБ і працює дуже швидко
        print("Loading Whisper model for evaluation...")
        self.whisper_model = whisper.load_model("base", device=self.device)

    def get_f0_pearson(self, f0_ref, f0_gen):
        """1. Метрика Інтонації (Мелодії)"""
        # Обрізаємо до мінімальної довжини
        min_len = min(len(f0_ref), len(f0_gen))
        ref = f0_ref[:min_len]
        gen = f0_gen[:min_len]
        
        # Рахуємо кореляцію ТІЛЬКИ там, де є голос (f0 > 0 в обох масивах), 
        # інакше нулі (тиша) сильно викривлять статистику
        mask = (ref > 0) & (gen > 0)
        
        if np.sum(mask) < 2: # Якщо збігів немає або масив надто малий
            return 0.0
            
        corr, _ = pearsonr(ref[mask], gen[mask])
        
        # Pearson може повернути NaN, якщо масив складається з однакових значень
        return 0.0 if np.isnan(corr) else corr

    def get_rms_pearson(self, audio_ref, audio_gen, sr=44100):
        """2. Метрика Динаміки (Гучність/Експресія)"""
        # Витягуємо гучність (RMS)
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
        
        # Завантажуємо аудіо через librosa (одразу в 16000 Гц, як того вимагає Whisper)
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000)
        gen_audio, _ = librosa.load(gen_audio_path, sr=16000)
        
        # Розпізнаємо текст напряму з масивів numpy
        ref_result = self.whisper_model.transcribe(ref_audio, language=language)
        ref_text = ref_result["text"].strip().lower()
        
        gen_result = self.whisper_model.transcribe(gen_audio, language=language)
        gen_text = gen_result["text"].strip().lower()
        
        if not ref_text:
            return 1.0 if gen_text else 0.0
            
        error_rate = jiwer.wer(ref_text, gen_text)
        return error_rate