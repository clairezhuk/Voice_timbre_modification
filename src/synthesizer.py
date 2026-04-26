import os
import sys
import yaml
import torch
import torch.nn.functional as F
import librosa
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ddsp_svc_path = os.path.join(project_root, "DDSP_SVC_6")
if ddsp_svc_path not in sys.path:
    sys.path.insert(0, ddsp_svc_path)

from DDSP_SVC_6.reflow.vocoder import Unit2Wav
from DDSP_SVC_6.reflow.vocoder import Vocoder 

class DictToDotDict(dict):
    def __getattr__(self, key):
        if key not in self:
            return None
        value = self[key]
        if isinstance(value, dict):
            return DictToDotDict(value)
        return value

class DDSPGenerator:
    def __init__(self, model_path, config_path, hifigan_path, device="cuda"):
        self.device = device
        self.model = self._load_model(model_path, config_path)
        self.vocoder = Vocoder("nsf-hifigan", hifigan_path, device)

    def _load_model(self, model_path, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        config = DictToDotDict(config)

        model = Unit2Wav(
            sampling_rate=config.data.sampling_rate or 44100,
            block_size=config.data.block_size or 512,
            win_length=config.model.win_length or 1024,
            n_unit=768, 
            n_spk=config.model.n_spk or 1,
            use_pitch_aug=config.model.use_pitch_aug or False,
            out_dims=config.model.out_dims or 128,
            n_layers=10,
            n_chans=1024
        )
        
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt['model'])
        
        return model.to(self.device).eval()

    def generate(self, content, f0, audio_44k, shift=0):
        hop_size = 512
        target_len = len(audio_44k) // hop_size
        
        # 1. F0 + Shift + Маска тиші
        f0_shifted = f0 * (2 ** (shift / 12))
        
        time_16k = np.linspace(0, 1, len(f0_shifted))
        time_44k = np.linspace(0, 1, target_len)
        uv_mask = (f0_shifted == 0).astype(float)
        
        f0_aligned = np.interp(time_44k, time_16k, f0_shifted)
        uv_aligned = np.interp(time_44k, time_16k, uv_mask) > 0.5
        f0_aligned[uv_aligned] = 0.0
        
        f0_pt = torch.from_numpy(f0_aligned).float().unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # 2. Гучність
        volume = librosa.feature.rms(y=audio_44k, frame_length=1024, hop_length=hop_size)[0]
        volume_aligned = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(volume)), volume)
        volume_pt = torch.from_numpy(volume_aligned).float().unsqueeze(0).unsqueeze(-1).to(self.device)

        # 3. Content
        content = F.interpolate(content, size=target_len, mode='linear', align_corners=False)
        content = content.transpose(1, 2).to(self.device)

        with torch.no_grad():
            # Передаємо vocoder і кажемо return_wav=True. 
            # Модель сама правильно згенерує mel і пропустить його через vocoder.
            generated_audio = self.model(
                content, f0_pt, volume_pt, 
                vocoder=self.vocoder, 
                infer=True, 
                return_wav=True, 
                infer_step=50, 
                method='euler'
            )
            
        return generated_audio.squeeze().cpu().numpy()