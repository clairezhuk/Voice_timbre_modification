import torch
import librosa
import numpy as np
from voice_clone_project.rmvpe.src.inference import RMVPE

class FeatureExtractor:
    def __init__(self, hubert_path, rmvpe_path, device="cuda"):
        self.device = device
        self.hubert = self._load_hubert(hubert_path)
        self.rmvpe = RMVPE(rmvpe_path)

    def _load_hubert(self, path):
        from fairseq import checkpoint_utils
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([path])
        model = models[0].to(self.device).eval()
        return model

    def extract_features(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        torch.cuda.empty_cache()
        
        audio_pt = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            units = self.hubert.extract_features(audio_pt, padding_mask=None)["x"]
            content = units.transpose(1, 2)

        f0 = self.rmvpe.infer_from_audio(audio, sample_rate=16000, device=self.device)
        
        return content, f0