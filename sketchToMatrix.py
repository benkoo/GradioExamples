import gradio as gr
import numpy as np

def display_matrix(img):
    # Get sketchpad image
    img = img['image']

    # Reshape to square matrix for display
    sq_dim = int(np.sqrt(img.size))
    img_matrix = img.reshape(sq_dim, sq_dim)

    return img_matrix

iface = gr.Interface(
    fn=display_matrix, 
    inputs=gr.Sketchpad(),
    outputs='label'
)

iface.launch()