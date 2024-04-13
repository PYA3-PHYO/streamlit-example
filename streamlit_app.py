import os
import cv2
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image 
import streamlit as st
from io import StringIO
from datetime import datetime
from ultralytics import YOLO

st.title("This is Aurora...")
st.title("Vehicle Type Detection Model :blue[1.0] :sunglasses:...")

st.caption("Please upload an image to  identify any Vehicle quickly and easily with our AI-Powered toolvehicles types..")
st.caption("Oue Model can predict 5 vehicle classes: Bus, car, Ambulance, Truck, and Motorcycle")
st.caption("Powered by advanced artificial intelligence YOLOv8...")

model  = YOLO("runs/best.pt")
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image")

if st.button("Detect.."):
    st.header("Here is your Result!", divider="rainbow")
    # 2) pass image to the model to get the detection result
    with st.spinner("Wait for it..."):
    
        result_img = model.predict(img, imgsz=640, conf=0.25)
        arry_img_result = result_img[0].plot()
 
        # # Predictions
        image_rgb = cv2.cvtColor(arry_img_result,cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Model Prediction(s)")
