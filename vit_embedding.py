from transformers import ViTImageProcessor, ViTModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Vit:
    def __init__(self):
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.eval()

    def get_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        return embedding
