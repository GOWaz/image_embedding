import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"


class Clip:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_embedding(self, img):
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings
