import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image

col1 = st.columns([2])

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input size of your model
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("ASL Finger Spelling Recognition")

# Upload image through the Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
camera_photo = st.camera_input("Take a photo")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make predictions using the pre-trained model
    #     predictions = model.predict(img_array)  - # model not ready for this line to function

    # Display the predicted class
    # predicted_class = np.argmax(predictions)
    # st.subheader(f"Predicted Class: {predicted_class}")