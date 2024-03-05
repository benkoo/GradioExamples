from typing import Dict, List
import gradio as gr
import numpy as np
import PIL as PIL
from PIL import Image

def convert_to_pil_image(image_data):
    """
    Convert different representations of images into a PIL (Python Imaging Library) image object.

    Args:
        image_data (any): The input image data that needs to be converted. It can be of type `Image.Image`, `str` (file path), or `np.ndarray` (NumPy array).

    Returns:
        PIL.Image: The converted PIL image object.

    Raises:
        ValueError: If the image representation is not supported.
    """
    if isinstance(image_data, Image.Image):
        return image_data

    if isinstance(image_data, str):
        return Image.open(image_data)

    if isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data)

    raise ValueError("Unsupported image representation")
    
def resize_to_pixel_count(image, pixel_count):
    width, height = image.size
    new_size = (pixel_count, pixel_count) 
    image = image.resize(new_size, Image.NEAREST)
    return image
    
def process_sketch(sketch_dict: Dict[str, any], pixel_count) -> Image.Image:
    """
    Process a sketch by combining the layers onto a background image.

    Args:
        sketch_dict (dict): A dictionary representing a sketch. It should have a "background" key with the value being the path to the background image file, and a "layers" key with the value being a list of paths to the layer image files.

    Returns:
        PIL.Image: The processed sketch image with the layers combined onto the background.

    Raises:
        ValueError: If the "background" key is not present in the sketch_dict.
    """
    if 'background' not in sketch_dict:
        raise ValueError("This is not sketchpad data.")

    background = convert_to_pil_image(sketch_dict["background"])

    for layer_path in sketch_dict["layers"]:
        layer_image = convert_to_pil_image(layer_path)
        background.paste(layer_image, (0, 0), layer_image)  # Paste with transparency

    newImg = resize_to_pixel_count(background, pixel_count)
    
    return [2, newImg]


iface = gr.Interface(
    fn=process_sketch, 
    inputs=[
        gr.Sketchpad(),
        gr.Slider(minimum=50, maximum=500, step=10)
    ],
    outputs=["text", gr.Image()],
)

iface.launch()