import torch
import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, hubert_path, rmvpe_path, device="cuda"):
        self.device = device
        
        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([hubert_path])
        self.hubert = models[0].to(device).eval()
        
        from DDSP_SVC_6.encoder.rmvpe.inference import RMVPE
        self.rmvpe = RMVPE(rmvpe_path)
        # Move model to GPU manually since __init__ no longer does it
        self.rmvpe.model = self.rmvpe.model.to(device)

    def extract_features(self, audio_path):
        audio_44k, _ = librosa.load(audio_path, sr=44100)
        audio_16k = librosa.resample(audio_44k, orig_sr=44100, target_sr=16000)
        
        torch.cuda.empty_cache()
        
        audio_pt = torch.from_numpy(audio_16k).unsqueeze(0).to(self.device)
        with torch.no_grad():
            units = self.hubert.extract_features(audio_pt, padding_mask=None)[0]
            content = units.transpose(1, 2)

        # Remove device and is_half args from inference call if they cause issues
        f0 = self.rmvpe.infer_from_audio(audio_16k, sample_rate=16000)
        
        return content, f0, audio_44k