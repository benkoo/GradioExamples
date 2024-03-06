import tensorflow as tf
import requests
import gradio as gr
import numpy as np
from PIL import Image

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def resize_image_pil(img, new_width, new_height):

    # Convert to PIL image
    img = Image.fromarray(img)
    
    # Get original size
    width, height = img.size

    # Calculate scale
    width_scale = new_width / width
    height_scale = new_height / height 
    scale = min(width_scale, height_scale)

    # Resize
    resized = img.resize((int(width*scale), int(height*scale)), Image.NEAREST)
    
    # Crop to exact size
    resized = resized.crop((0, 0, new_width, new_height))

    return resized

def classify_image(inp):
    img_resized = resize_image_pil(inp, 224, 224)
    arr = np.array(img_resized)
    reshaped_array = arr.reshape((-1, 224, 224, 3))
    reshaped_array = tf.keras.applications.mobilenet_v2.preprocess_input(reshaped_array)
    prediction = inception_net.predict(reshaped_array).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return [confidences, img_resized]

inception_net = tf.keras.applications.MobileNetV2()



gr.Interface(fn=classify_image,
             inputs=gr.Image(width=224, height=224),
             outputs=[gr.Label(num_top_classes=3), gr.Image()],
             examples=["images/lion.jpg", "images/truck.jpg"]).launch()