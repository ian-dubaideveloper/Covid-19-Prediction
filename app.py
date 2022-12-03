# importing the libraries
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time

# function to predict the uploaded image
def predict(image, classifier):
    image_shape = (256, 256, 3) # shape of the uploaded image
    test_image = image.resize((256,256)) # reshaping the uploaded image
    test_image = preprocessing.image.img_to_array(test_image) # converting the image to a numpy array
    test_image = test_image / 255.0 # normalizing the array values
    test_image = np.expand_dims(test_image, axis=0) # adding another dimension to the image array

    class_names = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia'] # class names
    predictions = classifier.predict(test_image) # doing prediction for the uploaded image
    scores = tf.nn.softmax(predictions[0]) # finding the score of prediction
    scores = scores.numpy()
    
    # string containing the prediction
    result = f"The uploaded image belongs to {class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} % confidence." 
    return result # returning thr string

# main function 
def main():
    st.header('Covid 19 detection using X-Ray images') # title for the page
    st.markdown('This project takes X-Ray images as input and predicts whether it belong to the following classes')
    st.markdown('Covid-19')
    st.markdown('Healthy')
    st.markdown('Lung Opacity')
    st.markdown('Viral Pneumonia')

    # loading the saved model
    model = load_model('model.h5', compile=False)#, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # uploading the image
    uploaded_image = st.file_uploader("Choose an image to be predicted", type=["png","jpg","jpeg"])
    

    if uploaded_image is not None: # if image is uploaded
        image_data = Image.open(uploaded_image) # opening the image
        st.image(image_data, caption = 'Uploaded X-Ray image', use_column_width=False) # displaying the image
        
        class_btn = st.button('Classify') # button to mention classify
        
        if class_btn: # if classify button is pressed
            if uploaded_image is None: # if no image is uploaded
                st.write('Please upload a valid image')
            else:
                with st.spinner("Classifying..."): # creating a spinner
                    plt.imshow(image_data) # displaying the image
                    plt.axis("off") # disabling the axis
                    predictions = predict(image_data, model) # calling the function to do the prediction
                    time.sleep(1)
                    st.success('Classified')
                    st.write(predictions)

if(__name__ == '__main__'):
    main()
