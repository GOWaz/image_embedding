import numpy as np
import cv2
from PIL import Image
import joblib
from scipy.cluster.vq import vq


class Bovw:
    def __init__(self):
        self.k, self.codebook = joblib.load("bovw-codebook.pkl")
        self.idf = np.load("idf.npy")
        self.sift = cv2.SIFT_create()

    def get_embedding(self, pil_image: Image.Image) -> np.ndarray:
        img_np = np.array(pil_image.convert("RGB"))
        img_np = cv2.resize(img_np, (224, 224))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype('uint8')

        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.k)  # return zero-vector if no features found

        visual_words, _ = vq(descriptors, self.codebook)

        freq_vector = np.zeros(self.k)
        for word in visual_words:
            freq_vector[word] += 1

        tfidf_vector = freq_vector * self.idf

        return tfidf_vector
