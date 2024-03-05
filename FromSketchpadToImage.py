import gradio as gr
import PIL as PIL
from PIL import Image
import numpy as np



# Helper function (you might need to adjust based on your representation)
def convert_to_pil_image(image_data):
    if isinstance(image_data, Image.Image):
        return image_data
    elif isinstance(image_data, str):  # Assuming filepath
        return Image.open(image_data)
    elif isinstance(image_data, np.ndarray):  
        return Image.fromarray(image_data)
    else:
        raise ValueError("Unsupported image representation")
    
def process_sketch(sketch_dict):

    if 'background' in sketch_dict:
        background = convert_to_pil_image(sketch_dict["background"])

        # Iterate through layers and paste them onto the background
        for layer in sketch_dict["layers"]:
            layer_image = convert_to_pil_image(layer)
            background.paste(layer_image, (0, 0), layer_image)  # Paste with transparency

        return background 
    else:
        raise ValueError("This is not sketchpad data.")
        

# Define the Gradio interface
interface = gr.Interface(fn=process_sketch,
                         inputs=gr.Sketchpad(),
                         outputs="image",
                         title="Sketch to Grayscale Converter",
                         description="Draw something and see it converted to grayscale!")

# Launch the app
interface.launch()