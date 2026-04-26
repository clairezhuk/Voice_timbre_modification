from src.feature_extractor import FeatureExtractor
from src.synthesizer import DDSPGenerator
from src.evaluator import Evaluator
import soundfile as sf

def main():
    device = "cuda"
    paths = {
        "hubert": "voice_clone_project/models/hubert_base.pt",
        "rmvpe": "voice_clone_project/models/model_0.pt",
        "ddsp": "voice_clone_project/models/model_80000.pt",
        "hifigan": "voice_clone_project/models/nsf_hifigan/model",
        "ddsp_config": "voice_clone_project/DDSP_SVC_6/configs/reflow.yaml"
    }

    # Ekstraction
    extractor = FeatureExtractor(paths["hubert"], paths["rmvpe"], device)
    content, f0, audio = extractor.extract_features("voice_clone_project/data/input/test2.wav")
        

    # Sintez + Vocoder
    generator = DDSPGenerator(paths["ddsp"], paths["ddsp_config"], paths["hifigan"], device)
    final_audio = generator.generate(content, f0, audio)

    sf.write("voice_clone_project/data/output/output_test.wav", final_audio.flatten(), 44100)
   
    # Evaluation
    evaluator = Evaluator()
    content_output, f0_output, audio_output = extractor.extract_features("voice_clone_project/data/output/output_test.wav")
    error = evaluator.check_pitch_accuracy(f0, f0_output) 
    quality =  evaluator.estimate_quality(audio_output)

    print(f"Test completed. Pitch error: {error:.4f}, quality: {quality:.4f}")

if __name__ == "__main__":
    main()