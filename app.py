import gradio as gr
from PIL import Image
import tensorflow.keras.models as KerasModel
import numpy as np

# Note that a tensorflow-based model called model.h5 must already been trained
model = KerasModel.load_model('model.h5')

# The first utility function for image resize
"""Resize an image to have a specified number of pixels along its width and height.

Args:
  image: The PIL Image object to resize.
  pixel_count: The number of pixels to resize the image to along its width and height.

Returns:
  A new PIL Image object containing the resized image.
"""


def resize_to_pixel_count(image, pixel_count):
    width, height = image.size
    new_size = (pixel_count, pixel_count)
    image = image.resize(new_size, Image.NEAREST)
    return image


# The second utility function for converting various kinds of data to PIL.Image
"""Converts various input types to PIL Image objects.

Supported input types:
- PIL Image objects (returned unchanged)  
- Filepaths (opened as PIL Image)
- Numpy arrays (converted with PIL.Image.fromarray)

Raises ValueError for unsupported types.
"""


def convert_to_pil_image(image_data):

    if isinstance(image_data, Image.Image):
        return image_data

    elif isinstance(image_data, str):  # Assuming filepath
        return Image.open(image_data)

    elif isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data)

    else:
        raise ValueError("Unsupported image representation")


# The third utility function for converting the sketchpad-generated data to image
"""Processes a sketchpad sketch dict into a PIL image.

The sketch dict is expected to have a 'background' key
with the background image data, and a 'layers' key containing
a list of layer images to composite on top of the background.

Raises a ValueError if the input does not look like valid 
sketchpad data.
"""


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


"""Recognizes a handwritten digit from a sketchpad sketch.

Converts the sketchpad sketch data into a PIL image, resizes it to 28x28 
pixels, and runs prediction on the alpha channel using the pre-trained
digit recognition model.

Returns a tuple containing the predicted digit probabilities and the 
preprocessed image.
"""


def recognize_digit(image):

    convertedFromSketchPad = None

    if 'background' in image:

        convertedFromSketchPad = convert_to_pil_image(image["background"])

        # Iterate through layers and paste them onto the background
        for layer in image["layers"]:
            layer_image = convert_to_pil_image(layer)
            convertedFromSketchPad.paste(layer_image, (0, 0), layer_image)

        img = resize_to_pixel_count(convertedFromSketchPad, 28)

        # Split the image into channels and extract the alpha channel
        r, g, b, alpha = img.split()

        # Convert the alpha channel to a numpy array, reshape, and normalize
        alpha_array = np.array(alpha).reshape(1, 28, 28, 1).astype('float32') / 255

        if alpha_array is not None:  # flatten to 1D
            prediction = model.predict(alpha_array)
            return [{str(i): float(prediction[0][i]) for i in range(10)}, img]

        else:
            return ['', img]


# The Gradio.Interface function
"""Creates a Gradio interface for digit recognition.

The interface takes a sketchpad input, runs digit recognition on it using the recognize_digit() function, 
and outputs the predicted digit probabilities and preprocessed image."""
iface = gr.Interface(
        fn=recognize_digit,
        inputs=gr.Sketchpad(),
        outputs=[gr.Label(num_top_classes=10), gr.Image()],
)

# Launch the Interface
"""Converts the sketchpad sketch data into a PIL image, resizes it to 28x28
pixels, and runs prediction on the alpha channel using the pre-trained
digit recognition model.

Returns a tuple containing the predicted digit probabilities and the
preprocessed image."""

iface.launch()
