import os
import torch
import streamlit as st

from PIL import Image
from io import BytesIO
from torchvision import transforms

# Define maximum file size
MAX_CONTENT_LENGTH = 2 * 1000 * 1000

# Load the pre-trained model
model = torch.load(os.path.join(os.path.dirname(__file__), "model.pt"))
model.eval()

# Define the input transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

st.set_page_config(page_title="Crack Detection App")

st.header(':wrench: Detect cracks from your images', divider='rainbow')
st.write(
    (
        ":dog: Try uploading an image to see whether it contains any cracks. "
        "This app uses a pre-trained Resnet18 model to detect cracks in images. "
        "The model was trained on a dataset of images containing cracks and non-cracks. "
        "The model is able to detect cracks with high accuracy. "
        "Full quality images can be downloaded from the sidebar. "
        "This code is open source and available [here](https://github.com/noodleslove/crack-detector) on GitHub. :grin:"
    )
)
st.sidebar.write("## Upload and download :gear:")


def convert_image(img):
    """Download the fixed image."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_img = buf.getvalue()
    return byte_img


def predict(upload):
    """Predict whether the image contains a crack."""
    img = Image.open(upload)
    st.write("Uploaded Image :camera:")
    st.image(img)
    input = transform(img).unsqueeze(0)

    # Make a prediction
    z = model(input)
    _, pred = torch.max(z, 1)

    # Display the prediction
    st.markdown("## Prediction :mag:\n")
    if pred.item() == 0:
        st.write("No cracks detected :white_check_mark:")
    else:
        st.write("Cracks detected :x:")

# Set up the file uploader
upload_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if the image is uploaded
if upload_img:
    if upload_img.size > MAX_CONTENT_LENGTH:
        st.error("Image too large. Please upload an image smaller than 2MB.")
    else:
        predict(upload_img)
else:
    st.write("Upload an image to get started :point_left:")
