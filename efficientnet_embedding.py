from torchvision import models, transforms
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class EfficientNet:
    def __init__(self):
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(device)
        self.model.classifier = torch.nn.Identity()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_embedding(self, image):
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding
