Here's a complete and professional-looking `README.md` for your image similarity project:

---

# 🧠 Image Embedding Similarity Explorer

This project allows users to upload a query image and compare it against a set of other images using various image embedding models. The results are sorted based on cosine similarity—from most similar to least similar. Built with **Python** and **Gradio**, and hosted on **Hugging Face Spaces**.

🔗 **Live Demo**: [Check it out on Hugging Face](https://huggingface.co/spaces/GOWaz/Image_Embedding)

## 🚀 Features

- Upload a query image and multiple target images.
- Choose between different embedding models.
- Get ranked similarity scores using **cosine similarity**.
- Intuitive **Gradio-based UI**.
- Hosted on Hugging Face for easy sharing.

## 🧰 Models Used

- **CLIP** (Image-only)
- **DINO**
- **EfficientNet**
- **ViT Transformer**
- **Histogram-based comparison**
- **Bag of Visual Words (BoVW)**

## 📦 Local Setup Instructions

Follow the steps below to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/GOWaz/image_embedding.git
cd your-repo-name
```

### 2. Create a Python Virtual Environment

```bash
python -m venv .env
```

### 3. Activate the Environment

- On Windows:

  ```bash
  .env\Scripts\activate
  ```

- On macOS/Linux:

  ```bash
  source .env/bin/activate
  ```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
python app.py
```

The Gradio interface should launch locally in your browser. You can now start uploading images and exploring similarity comparisons.

## 🧠 How It Works

1. **Embedding Extraction**: The selected model converts each image into a high-dimensional vector (embedding).
2. **Similarity Calculation**: Cosine similarity is calculated between the query image and each target image.
3. **Sorting**: The results are sorted in descending order of similarity.

## 🌐 Live Preview

[![Hugging Face Spaces](https://img.shields.io/badge/🤗_View%20on-Hugging%20Face-orange?style=for-the-badge)](https://huggingface.co/spaces/GOWaz/Image_Embedding)
