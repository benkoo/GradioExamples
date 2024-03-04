
import tensorflow as tf

import requests
import numpy as np
import cv2

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

inception_net = tf.keras.applications.MobileNetV2()


def classify_image(inp):
    inp = cv2.resize(inp, (224, 224))  # Ensure image is 224x224

    # If the image is grayscale, convert to RGB (this depends on your model's expectation)
    if len(inp.shape) == 2:
        inp = cv2.cvtColor(inp, cv2.COLOR_GRAY2RGB)
        
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences

import gradio as gr

gr.Interface(fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    examples=["images/cats.jpg", "images/lion.jpg", "images/tower.jpg", "images/Dogs.jpeg", "images/cheetah1.jpg"]).launch()
