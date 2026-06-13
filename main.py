from src.feature_extractor import FeatureExtractor
from src.synthesizer import DDSPGenerator
from src.evaluator import Evaluator
import soundfile as sf
from DDSP.ddsp.vocoder import Units_Encoder, F0_Extractor
import torch
import librosa
import os

def main(NAME = "1_T_5_Characters-01"):
    N = 21500
    K_STEP = 500
    device = "cuda"
    paths = {
        "hubert": "voice_clone_project/models/contentvec768l12.pt",
        "rmvpe": "voice_clone_project/models/model_0.pt",
        "ddsp": f"voice_clone_project/models/model_{N}.pt",
        "hifigan": "voice_clone_project/models/nsf_hifigan/model",
        "ddsp_config": 'voice_clone_project/my_model/config.yaml' #"voice_clone_project/DDSP_SVC_6/configs/reflow.yaml"
    }
    SHIFT = 0
    TEST = 1
    INPUT_PATH = f"voice_clone_project/data/dataset/test/input/{NAME}.wav"
    OUTPUT_PATH = f"voice_clone_project/data/dataset/test/output/{N}_{NAME}.wav"
    RES_PATH = f"voice_clone_project/data/dataset/test/output/metrics/{NAME}.csv"

    # Ekstraction
    extractor = FeatureExtractor(paths["hubert"], paths["rmvpe"], device)
    audio_44k, _ = librosa.load(INPUT_PATH, sr=44100)
    audio_pt_44k = torch.from_numpy(audio_44k).float().unsqueeze(0).to(device)

    encoder = Units_Encoder(
        encoder='contentvec768l12', 
        encoder_ckpt=paths["hubert"], 
        device=device
    )  
    units = encoder.encode(audio_pt_44k, sample_rate=44100, hop_size=512)
    f0_extractor = F0_Extractor(
        f0_extractor='rmvpe', 
        sample_rate=44100,      
        hop_size=512,           
        f0_min=65, 
        f0_max=800
    )
    f0 = f0_extractor.extract(audio_44k, uv_interp=True, device=device)

    # Sintez + Vocoder
    generator = DDSPGenerator(paths["ddsp"], paths["ddsp_config"], paths["hifigan"], device)
    final_audio = generator.generate(units, f0, audio_44k, k_step=K_STEP, shift=SHIFT)

    sf.write(OUTPUT_PATH, final_audio.flatten(), 44100)
   
    # Evaluation
    evaluator = Evaluator(device)
    audio_44k_out, _ = librosa.load(OUTPUT_PATH, sr=44100)
    audio_16k_out = librosa.resample(audio_44k_out, orig_sr=44100, target_sr=16000)
    f0_output = f0_extractor.extract(audio_16k_out, uv_interp=True, device=device)
    
    min_len_f0 = min(len(f0), len(f0_output))
    f0_pearson = evaluator.get_f0_pearson(f0[:min_len_f0], f0_output[:min_len_f0])
    
    min_len_audio = min(len(audio_44k), len(final_audio))
    rms_pearson = evaluator.get_rms_pearson(audio_44k[:min_len_audio], final_audio[:min_len_audio], sr=44100)
    
    wer_score = evaluator.get_wer(INPUT_PATH, OUTPUT_PATH)

    print(f"Test completed.")
    print(f"F0 Pearson Corr (Closer to 1 is better): {f0_pearson:.4f}")
    print(f"RMS Pearson Corr (Closer to 1 is better): {rms_pearson:.4f}")
    print(f"WER (Closer to 0 is better): {wer_score:.4f}")

    os.makedirs(os.path.dirname(RES_PATH), exist_ok=True)
    file_exists = os.path.isfile(RES_PATH)
    with open(RES_PATH, mode='a', encoding='utf-8') as f:
        if not file_exists:
            f.write("N,K_STEP,F0,RMS,WER\n")
        f.write(f"{N},{K_STEP},{f0_pearson:.4f},{rms_pearson:.4f},{wer_score:.4f}\n")

if __name__ == "__main__":
    names = ["S_6_Kickapoo-15","Queen-01",
             "Bring_Me_To_Life-01", "It`s_over_Anakin-01","Janet-01","Never_gona_give_you_up-01","Surprise-01","What_are_we_going_to_do-01"]
    for name in names:
        main(name)