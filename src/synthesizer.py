import os
import sys
import yaml
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ddsp_svc_path = os.path.join(project_root, "DDSP-SVC-5.0")
if ddsp_svc_path not in sys.path:
    sys.path.append(ddsp_svc_path)

from DDSP_SVC_6.reflow.vocoder import Reflow

class DDSPGenerator:
    def __init__(self, model_path, config_path, device="cuda"):
        self.device = device
        self.model = self._load_model(model_path, config_path)

    def _load_model(self, model_path, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        from dict_to_dotdict import DictToDotDict
        config = DictToDotDict(config) # DDSP-SVC очікує dot-нотацію

        # Будуємо каркас Reflow
        model = Reflow(
            config.model.out_dims,
            config.model.n_layers,
            config.model.n_chans,
            config.model.n_hidden
        )
        
        # Завантажуємо ваги
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt['model'])
        
        return model.to(self.device).eval()

    def generate(self, content, f0, shift=0):
        f0_shifted = f0 * (2 ** (shift / 12))
        f0_pt = torch.from_numpy(f0_shifted).float().unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Для Reflow моделі необхідний параметр steps (infer_step) - стандартно 10-20
        # метод 'euler' є найшвидшим
        with torch.no_grad():
            audio = self.model(content, f0_pt, infer_step=20, method='euler')
            
        return audio.squeeze().cpu().numpy()

# Допоміжний клас для конвертації словника yaml, щоб до нього можна було звертатись як config.model (додай його в кінець файлу або на початок)
class DictToDotDict(dict):
    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DictToDotDict(value)
        return value