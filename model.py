import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
# from model import ImageCaptioningModel  # Import your image captioning model

import tensorflow as tf
## Assuming you have the model architecture code
# from model import ImageCaptioningModel  # Import your model definition
from model import ImageCaptioningModel
model = ImageCaptioningModel(input_shape=(224, 224))  # Recreate the model architecture

# Load the weights from the saved file
model.load_weights('C:/Users/Daveee/Desktop/image caption model with app/weights_model.h5')

# Define transformations for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Function to generate caption
def generate_caption(image):
    image = transform(image).unsqueeze(0)
    caption = model.generate_caption(image)  # Implement this method in your model
    return caption

# Streamlit app
def main():
    st.title("Image Captioning App")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Generate Caption'):
            caption = generate_caption(image)
            st.write('Caption:', caption)

if __name__ == "__main__":
    main()
