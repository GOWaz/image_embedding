import os

import gradio as gr
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from clip_embedding import Clip
from efficientnet_embedding import EfficientNet
from vit_embedding import Vit
from resnet_embedding import Resnet
from dino_embedding import Dino
from histogram_embedding import cosine, get_embedding
from bovw_embedding import Bovw

resnet = Resnet()
vit = Vit()
efficientnet = EfficientNet()
bovw = Bovw()
dino = Dino()
clip = Clip()


def get_image_embedding(image: Image.Image, name):
    match name:
        case "ResNet":
            return resnet.get_embedding(image).cpu().numpy()
        case "VIT":
            return vit.get_embedding(image).cpu().numpy()
        case "EfficientNet":
            return efficientnet.get_embedding(image).cpu().numpy()
        case "Histogram":
            return get_embedding(image)
        case "BOVW":
            return bovw.get_embedding(image)
        case "DINO":
            return dino.get_embedding(image).cpu().numpy()
        case _:
            return clip.get_embedding(image).cpu().numpy()



def compare_images(main_img, compare_imgs, name):
    results = []
    if name in ("Histogram", "ResNet", "BOVW"):
        main_emb = get_image_embedding(main_img, name)
        for img in compare_imgs:
            emb = get_image_embedding(img, name)
            results.append((img, round(cosine(main_emb, emb) * 100, 2)))
    else:
        main_embedding = get_image_embedding(main_img, name)
        for img in compare_imgs:
            emb = get_image_embedding(img, name)
            score = cosine_similarity(main_embedding, emb)[0][0]
            percentage = round(score * 100, 2)
            results.append((img, percentage))


    results.sort(key=lambda x: x[1], reverse=True)

    return results


model_list = ["CLIP", "VIT", "EfficientNet", "ResNet", "DINO", "Histogram", "BOVW"]


with gr.Blocks() as demo:
    gr.Tab("Image Embedding")
    gr.Markdown("# Image Similarity Finder")
    gr.Markdown(
        "Upload a main image and compare it to others. Results show similarity percentages using embeddings.")

    with gr.Row():
        with gr.Column():
            main_image = gr.Image(type="pil", label="Main Image")
            compare_images_input = gr.File(file_count="multiple", file_types=["image"], label="Comparison Images")
            modelName = gr.Dropdown(model_list, label="Model", value=model_list[0])
            submit_btn = gr.Button("Compare")

        with gr.Column():
            gallery = gr.Gallery(label="Similarity Results")
            similarity_text = gr.Textbox(label="Similarity Scores")


    def process_comparison(main_img, compare_files, name):
        compare_imgs = [Image.open(file.name) for file in compare_files]
        results = compare_images(main_img, compare_imgs, name)

        # Prepare outputs
        images = [result[0] for result in results]
        scores = [f"Image: {os.path.basename(result[0].filename)} -> Similarity: {result[1]:.2f}%" for result in
                  results]

        return images, "\n".join(scores)


    submit_btn.click(
        fn=process_comparison,
        inputs=[main_image, compare_images_input, modelName],
        outputs=[gallery, similarity_text]
    )

demo.launch()
