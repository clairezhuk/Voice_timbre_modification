import torch

class Vocoder:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self._load_vocoder(model_path)

    def _load_vocoder(self, path):
        # HiFi-GAN 
        model = torch.jit.load(path, map_location=self.device)
        return model.eval()

    def render(self, spec):
        with torch.no_grad():
            return self.model(torch.from_numpy(spec).to(self.device)).cpu().numpy()