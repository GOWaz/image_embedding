import cv2
import numpy as np


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(img, bins=32):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (224, 224))

    blue = cv2.calcHist([img], [0], None, [bins], [0, 256])
    green = cv2.calcHist([img], [1], None, [bins], [0, 256])
    red = cv2.calcHist([img], [2], None, [bins], [0, 256])
    vector = np.concatenate([blue, green, red], axis=0)
    vector = vector.flatten()  # Flatten to 1D array

    return vector
