import gradio as gr
import tensorflow as tf 
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('model.h5')


def resize_to_pixel_count(image, pixel_count):
    width, height = image.size
    new_size = (pixel_count, pixel_count) 
    image = image.resize(new_size, Image.NEAREST)
    return image


def recognize_digit(image):
    convertedFromSketchPad = None
    if 'background' in image:
        convertedFromSketchPad = convert_to_pil_image(image["background"])

        # Iterate through layers and paste them onto the background
        for layer in image["layers"]:
            layer_image = convert_to_pil_image(layer)
            convertedFromSketchPad.paste(layer_image, (0, 0), layer_image)  

    low_res = resize_to_pixel_count(convertedFromSketchPad, 28)
    image = np.array(low_res, dtype='float32')
    
    if image is not None:  # flatten to 1D
        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255.0
        prediction = model.predict(image)
        return [{str(i): float(prediction[0][i]) for i in range(10)}, low_res]
    else:
        return ['', low_res]
    

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

        
iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(),
    outputs=[gr.Label(num_top_classes=10), gr.Image()],
)

iface.launch()
