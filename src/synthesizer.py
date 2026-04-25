import os
import sys
import yaml
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ddsp_svc_path = os.path.join(project_root, "DDSP_SVC_6")
if ddsp_svc_path not in sys.path:
    sys.path.insert(0, ddsp_svc_path)

from DDSP_SVC_6.reflow.vocoder import Unit2Wav

class DictToDotDict(dict):
    def __getattr__(self, key):
        if key not in self:
            return None
        value = self[key]
        if isinstance(value, dict):
            return DictToDotDict(value)
        return value

class DDSPGenerator:
    def __init__(self, model_path, config_path, device="cuda"):
        self.device = device
        self.model = self._load_model(model_path, config_path)

    def _load_model(self, model_path, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        config = DictToDotDict(config)

        model = Unit2Wav(
            sampling_rate=config.data.sampling_rate or 44100,
            block_size=config.data.block_size or 512,
            win_length=config.model.win_length or 1024,
            n_unit=256, 
            n_spk=config.model.n_spk or 1,
            use_pitch_aug=config.model.use_pitch_aug or False,
            out_dims=config.model.out_dims or 128,
            n_layers=config.model.n_layers or 6,
            n_chans=config.model.n_chans or 512
        )
        
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt['model'])
        
        return model.to(self.device).eval()

    def generate(self, content, f0, shift=0):
        f0_shifted = f0 * (2 ** (shift / 12))
        f0_pt = torch.from_numpy(f0_shifted).float().unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            audio = self.model(content, f0_pt, infer_step=20, method='euler')
            
        return audio.squeeze().cpu().numpy()