import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class Resnet:
    def __init__(self):
        self.model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).to(device)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_embedding(self, image):
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze()
        return embedding
