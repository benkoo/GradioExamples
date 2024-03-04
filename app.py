import gradio as gr
import tensorflow as tf 
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('model.h5')


def resize_image(image, new_width, new_height):
    resized_image = image.resize((new_width, new_height))
    return resized_image

def recognize_digit(image):
    convertedFromSketchPad = None
    if 'background' in image:
        convertedFromSketchPad = convert_to_pil_image(image["background"])

        # Iterate through layers and paste them onto the background
        for layer in image["layers"]:
            layer_image = convert_to_pil_image(layer)
            convertedFromSketchPad.paste(layer_image, (0, 0), layer_image)  

    low_res = resize_image(convertedFromSketchPad, 28, 28)
    print("The low_res image size is: " + str(low_res.size))
    low_res = low_res.convert('L') 
    img_array = np.array(low_res)
    
    if img_array is not None:  # flatten to 1D
        image = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255
        prediction = model.predict(image)
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return ''
    

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
    fn = recognize_digit,
    inputs = gr.Sketchpad(),
    outputs = gr.Label(num_top_classes=3),
)

iface.launch()