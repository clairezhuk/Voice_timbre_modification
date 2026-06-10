from src.feature_extractor import FeatureExtractor
from src.synthesizer import DDSPGenerator
from src.evaluator import Evaluator
import soundfile as sf

def main():
    device = "cuda"
    paths = {
        "hubert": "voice_clone_project/models/contentvec768l12.pt",
        "rmvpe": "voice_clone_project/models/model_0.pt",
        "ddsp": "voice_clone_project/models/model_26000.pt",
        "hifigan": "voice_clone_project/models/nsf_hifigan/model",
        "ddsp_config": 'voice_clone_project/my_model/config.yaml' #"voice_clone_project/DDSP_SVC_6/configs/reflow.yaml"
    }
    SHIFT = 12
    TEST = 1
    INPUT_PATH = f"voice_clone_project/data/dataset/test/input/1_...Baby One More Time_(Vocals).wav"
    OUTPUT_PATH = f"voice_clone_project/data/dataset/test/output/1_...Baby One More Time_(Vocals).wav"

    # Ekstraction
    extractor = FeatureExtractor(paths["hubert"], paths["rmvpe"], device)
    content, f0, audio_44k = extractor.extract_features(INPUT_PATH)
        

    # Sintez + Vocoder
    generator = DDSPGenerator(paths["ddsp"], paths["ddsp_config"], paths["hifigan"], device)
    final_audio = generator.generate(content, f0, audio_44k, shift=SHIFT)

    sf.write(OUTPUT_PATH, final_audio.flatten(), 44100)
   
    # Evaluation
    evaluator = Evaluator(device)
    content_output, f0_output, audio_output = extractor.extract_features(OUTPUT_PATH)
    
    min_len_f0 = min(len(f0), len(f0_output))
    f0_pearson = evaluator.get_f0_pearson(f0[:min_len_f0], f0_output[:min_len_f0])
    
    min_len_audio = min(len(audio_44k), len(final_audio))
    rms_pearson = evaluator.get_rms_pearson(audio_44k[:min_len_audio], final_audio[:min_len_audio], sr=44100)
    
    wer_score = evaluator.get_wer(INPUT_PATH, OUTPUT_PATH)

    print(f"Test completed.")
    print(f"F0 Pearson Corr (Closer to 1 is better): {f0_pearson:.4f}")
    print(f"RMS Pearson Corr (Closer to 1 is better): {rms_pearson:.4f}")
    print(f"WER (Closer to 0 is better): {wer_score:.4f}")

if __name__ == "__main__":
    main()