import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO
from ultralytics import YOLO
import cv2
from PIL import Image 

st.title('This is Aurora...')
st.title('Vehicle Type detection Model :blue[1.0] :sunglasses:...')

st.caption('Please upload an image to  identify any Vehicle quickly and easily with our AI-Powered toolvehicles types..')
st.caption('Oue Model can predict 5 vehicle classes: Bus, car, Ambulance, Truck, and Motorcycle')
st.caption('Powered by advanced artificial intelligence YOLOv8...')

# function to convert file buffer to cv2 image
def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    return cv2.imread(img_stream, cv2_img_flag)

model  = YOLO('runs/best.pt')
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # 1) image will converted to cv2 image
    str_img_file = str(uploaded_file)
    open_cv_image = create_opencv_image_from_stringio(str_img_file)



if st.button('Detect..'):
    st.header('Here is your Result!', divider='rainbow')
    # 2) pass image to the model to get the detection result
    result_img = model(open_cv_image, imgsz=320, conf=0.5)
    # # 3) extract numpy array from the list
    # result_array = result_img[0].numpy()
    # # 4) convert array to PIL image
    # pil_image = Image.fromarray(result_img)
    

  
    if result_img is not None:
        with st.spinner('Wait for it...'):
            st.image(result_img, channels="BGR")


