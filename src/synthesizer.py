import torch
import torch.nn as nn

class DDSPGenerator:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        model = torch.jit.load(path, map_location=self.device)
        return model.eval()

    def generate(self, content, f0, shift=0):
        # f0_shift 
        f0_shifted = f0 * (2 ** (shift / 12))
        f0_pt = torch.from_numpy(f0_shifted).float().unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            # Модель генерує аудіо на основі контенту та тону 
            audio = self.model(content, f0_pt)
        return audio.cpu().numpy()