import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
#from tensorflow.keras.models import load_model
from keras.models import load_model

import requests 
from urllib.parse import urljoin

"""******************"""

def get_image_dimensions(directory):
    dimensions = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path =  urljoin(root, file) # os.path.join(root, file)
                with Image.open(img_path) as img:
                    dimensions.append(img.size)
    return dimensions

def main():
    # Streamlit UI
    st.title('ASL Gesture Recognition App',)

    text = '''Speech impaired people use hand signs and gestures to communicate. Normal people face difficulty in understanding their language. Hence there is a need of a system which recognizes the different signs, gestures and conveys the information to the normal people. 
            
            
    This is the Example of Multiclass classification problem.
            
    This webapp includes loading the model to make predicitions on Hand signs.'''
    st.markdown(text)

    st.header("Data Description")
    text2 = '''The data set is a collection of images of alphabets including Space and Nothing from the American Sign Language, separated in 28 folders which represent the various classes.
    
    The data set contains images which are 400x400 pixels. There are 28 classes.
    
    The 28 classes contain A-Z alphabets, Space and Nothing.'''

    st.markdown(text2)

    image = Image.open('Mid_Term/Asl_Detection/hand_sign.png')

    st.image(image, caption='Hand signs')

    
    
    st.header("Image Dataset EDA and IDA")

    # Get dataset path from user input
    #       data_directory = st.text_input("Enter path to image dataset:")
    data_directory = "https://github.com/vipuljain-vinyl/CMSE-830/tree/main/Mid_Term/Asl_Detection/" #"""""********"""

    if not data_directory:
        st.warning("Please enter a valid path.")
        return

    # Load data
    train_data_dir = os.path.join(data_directory, 'Train_split')
    test_data_dir = os.path.join(data_directory, 'Test')

    # Check if the dataset path is valid
    """ ***************"""
    # if not (os.path.exists(train_data_dir) and os.path.exists(test_data_dir)):  
    #     st.warning("Invalid dataset path. Please check the path and try again.")
    #     return

    
    image_path_train = "Mid_Term/Asl_Detection/5.jpg"
    image_path_test = "Mid_Term/Asl_Detection/3004.jpg"
    
    dimensions_train = []
    with Image.open(image_path_train) as img:
                    dimensions_train.append(img.size)

    dimensions_test = []
    with Image.open(image_path_test) as img:
                    dimensions_test.append(img.size)

                
    train_dimensions = dimensions_train
    test_dimensions = dimensions_test
    


    train_df = pd.DataFrame(train_dimensions, columns=['Width', 'Height'])
    test_df = pd.DataFrame(test_dimensions, columns=['Width', 'Height'])

    # EDA
    st.header("Exploratory Data Analysis (EDA)")

    # Display summary statistics
    st.subheader("Training Data Summary:")
    st.write(train_df.describe())

    st.subheader("Testing Data Summary:")
    st.write(test_df.describe())

    #fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot = sns.jointplot(x='Width', y='Height', data=train_df, kind='scatter')
    st.pyplot(plot.fig)

    # Plot image dimensions distribution
    st.subheader("Image Dimensions Distribution:")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x='Width', y='Height', data=train_df, ax=ax[0])
    ax[0].set_title('Training Image Dimensions')
    sns.scatterplot(x='Width', y='Height', data=test_df, ax=ax[1])
    ax[1].set_title('Testing Image Dimensions')
    st.pyplot(fig)

    # IDA
    st.header("Interactive Data Analysis (IDA)")

    # # Set specific ranges for count plots
    # width_range = st.slider("Select width range:", 0, 200, (0, 200))
    # height_range = st.slider("Select height range:", 0, 200, (0, 200))


    # Plot class distribution

    # train_classes = [cls for cls in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, cls))]
    # test_classes = [cls for cls in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, cls))]
    train_classes=[]
    test_classes = []


    response = requests.get(train_data_dir)
    

    
    data = response.json()
    
    if 'payload' in data and 'tree' in data['payload']:
        items_data = data['payload']['tree']['items']
        
        for item in items_data:
            train_classes.append(item['name'])
    else:
        print("Tree information not found in the response.")
    

    


    response = requests.get(test_data_dir)
    

    
    data = response.json()
    
    if 'payload' in data and 'tree' in data['payload']:
        items_data = data['payload']['tree']['items']
        
        for item in items_data:
            test_classes.append(item['name'])
    else:
        print("Tree information not found in the response.")

    


    # Plot class distribution with specific ranges
    st.subheader("Class Distribution:")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.countplot(y = train_classes, ax=ax[0])
    ax[0].set_title('Training Class Distribution')
    sns.countplot(y = test_classes, ax=ax[1])
    ax[1].set_title('Testing Class Distribution')
    st.pyplot(fig)


    # make sure the number of labels matches
    labels = train_classes #os.listdir(train_data_dir)
    N=[]
    for i in range(len(labels)):
        N+=[i]
        
    mapping=dict(zip(labels,N)) 
    reverse_mapping=dict(zip(N,labels)) 

    def mapper(value):
        return reverse_mapping[value]
    
    
    

    # Load the trained model
    # model_file = st.file_uploader("choose trained model")
    # if model_file is not None:
    #     model = load_model(model_file)
    model =  load_model('Mid_Term/Asl_Detection/model_pretrained.h5')   

    uploaded_file = st.file_uploader("Choose an image for recognizing sign language...")

    if uploaded_file is not None:
        # Display the uploaded image
        image_file = load_img(uploaded_file, target_size=(40,40))
        st.image(image_file, caption="Uploaded Image.", use_column_width=True)

        image_file=img_to_array(image_file) 
        image_file=image_file/255.0
        # pred_image=np.array(image_file)
        pred_image= np.expand_dims(image_file, axis=0)

        # Make prediction
        prediction=model.predict(pred_image)
        value=np.argmax(prediction)
        sign_name=mapper(value)
        st.write(f"Prediction: {sign_name}")



if __name__ == "__main__":
    main()
