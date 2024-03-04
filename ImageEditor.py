import gradio as gr
import PIL.Image

def edit_image(image):
    # ImageEditor passes image as PIL Image object
    img = PIL.Image.fromarray(image['layers'])
    edited = PIL.Image.fromarray(img)
    
    # Do some sample editing 
    edited = edited.rotate(90)
    edited = edited.convert('L')
    
    return edited

iface = gr.Interface(
    fn=edit_image,
    inputs=gr.ImageEditor(),
    outputs="image",
    title="Image Editor",
    description="Edit an image and see the result",
)

if __name__ == "__main__":
    iface.launch()