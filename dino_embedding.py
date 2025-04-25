import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class Dino:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def get_embedding(self, image):
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding
