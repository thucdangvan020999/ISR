import gradio
from huggingface_hub import hf_hub_download
import onnxruntime
from PIL import Image
import numpy as np

path = hf_hub_download("xiongjie/lightweight-real-ESRGAN-anime", filename="RealESRGAN_x4plus_anime_4B32F.onnx")
session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

def upscale(np_image_rgb):
    # From RGB to BGR
    np_image_bgr = np_image_rgb[:, :, ::-1]
    np_image_bgr = np_image_bgr.astype(np.float32)
    np_image_bgr /= 255
    np_image_bgr = np.transpose(np_image_bgr, (2, 0, 1))
    np_image_bgr = np.expand_dims(np_image_bgr, axis=0)
    output_img = session.run([], {"image.1":  np_image_bgr})[0]
    output_img = np.squeeze(output_img, axis=0).astype(np.float32).clip(0, 1)
    output_img = np.transpose(output_img, (1, 2, 0))
    output = (output_img * 255.0).astype(np.uint8)
    # From BGR to RGB
    output = output[:, :, ::-1]
    
    return output

import os

import cv2
import streamlit as st
from PIL import Image

import time


from PIL import Image



upload_path = "/Users/dx/Documents/VSCODE/Streamlit-based-Image-Super-Resolution-using-ESRGAN/uploads/"
download_path = "/Users/dx/Documents/VSCODE/Streamlit-based-Image-Super-Resolution-using-ESRGAN/downloads/"
main_image = Image.open('/Users/dx/Documents/VSCODE/Streamlit-based-Image-Super-Resolution-using-ESRGAN/static/compare.png')
st.set_page_config(
    page_title="ISR using ESRGAN",
    page_icon="💫",
    layout="centered",
    initial_sidebar_state="auto",
)
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

activities = ["Compare between SRGAN and ESRGAN", "ISR for Face", "ISR for ANIME"]
choice = st.sidebar.selectbox("Please choose an actin", activities)
if choice == "Compare between SRGAN and ESRGAN":

    st.image(main_image,use_column_width='auto')
    st.title("👨‍💻👨‍💻IMAGE SUPER RESOLUTION👨‍💻👨‍💻") 
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


    add_bg_from_local('/Users/dx/Documents/VSCODE/Streamlit-based-Image-Super-Resolution-using-ESRGAN/static/background.png')    

    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image 🚀", type=["png","jpg","bmp","jpeg"])
    print(upload_path,uploaded_file)    
    if uploaded_file is not None:
            with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
                
                f.write((uploaded_file).getbuffer())
            with st.spinner(f"Working... 💫"):
                uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
                st.image(uploaded_image, caption='This is the original image 😉')
                image_ = Image.open(uploaded_image)
                image_anime = upscale(image_)
                print(image_anime)

            
    else:
        st.warning('⚠ Please upload your Image file 😯')


    st.markdown("<br><hr><center>Made with ❤️ by <a href='mailto:thucdangvan020999@gmail.com?subject=ISR using ESRGAN WebApp!&body=Please specify the issue you are facing with the app.'><strong>ЛЫУ ТИЕН ДАТ 👱‍♂️ КМБО-03-19 </strong></a></center><hr>", unsafe_allow_html=True)

