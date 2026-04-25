from src.feature_extractor import FeatureExtractor
from src.synthesizer import DDSPGenerator
from src.vocoder import Vocoder
from src.evaluator import Evaluator
import soundfile as sf

def main():
    device = "cuda"
    paths = {
        "hubert": "voice_clone_project/models/hubert_base.pt",
        "rmvpe": "voice_clone_projec/models/rmvpe.pt",
        "ddsp": "voice_clone_project/models/ddsp_default.pt",
        "hifigan": "voice_clone_project/models/nsf_hifigan.pt"
    }

    # 1. Екстракція
    extractor = FeatureExtractor(paths["hubert"], paths["rmvpe"], device)
    content, f0 = extractor.extract_features("voice_clone_project/data/input/test_audio.wav")
    
    # 2. Синтез (використовуємо дефолтний голос)
    generator = DDSPGenerator(paths["ddsp"], device)
    raw_audio = generator.generate(content, f0)
    
    # 3. Вокодер
    vocoder = Vocoder(paths["hifigan"], device)
    final_audio = vocoder.render(raw_audio)
    
    # 4. Оцінка
    evaluator = Evaluator()
    error = evaluator.check_pitch_accuracy(f0, f0) # Тест на ідентичність
    
    sf.write("voice_clone_project/data/output/output_test.wav", final_audio.flatten(), 16000)
    print(f"Test completed. Pitch error: {error:.4f}")

if __name__ == "__main__":
    main()