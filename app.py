


import os
from brisque import BRISQUE
from PIL import Image, ImageEnhance
import cv2
import streamlit as st
from PIL import Image
from app_funcs import *
from app_func_face import *
from streamlit_image_comparison import image_comparison
import time
from sharpening_images.sharpening import sharpening_image

import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr


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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = RealESRGAN(device, scale=2)
model2.load_weights('models/RealESRGAN_x2.pth', download=True)
model4 = RealESRGAN(device, scale=4)
model4.load_weights('models/RealESRGAN_x4.pth', download=True)
model8 = RealESRGAN(device, scale=8)
model8.load_weights('models/RealESRGAN_x8.pth', download=True)

def inference(image, size):
    if size == '2x':
        result = model2.predict(image.convert('RGB'))
    elif size == '4x':
        result = model4.predict(image.convert('RGB'))
    else:
        result = model8.predict(image.convert('RGB'))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


upload_path = "uploads/"
download_path = "downloads/"
main_image = Image.open('static/compare.png')
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


    add_bg_from_local('static/background.png')    

    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image 🚀", type=["png","jpg","bmp","jpeg"])

    if uploaded_file is not None:
            with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
                f.write((uploaded_file).getbuffer())
            with st.spinner(f"Working... 💫"):
                uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
                st.image(uploaded_image, caption='This is the original image 😉')
                
                if st.button('COMPARE 2 ALGORITHMS'):
                    downloaded_image_esrgan = os.path.abspath(os.path.join(download_path,str("output_1"+uploaded_file.name)))
                    downloaded_image_srgan = os.path.abspath(os.path.join(download_path,str("output_2"+uploaded_file.name)))

                    model_esrgan = instantiate_model('ESRGAN model ✅')
                    start_esrgan = time.time()
                    image_super_resolution(uploaded_image, downloaded_image_esrgan, model_esrgan)
                    final_image_esrgan = Image.open(downloaded_image_esrgan)
                    final_esrgan = time.time()
                    execution_time_esrgan = final_esrgan-start_esrgan

                    model_srgan = instantiate_model('SRGAN model ✅')
                    start_srgan = time.time()
                    image_super_resolution(uploaded_image, downloaded_image_srgan, model_srgan)
                    final_image_srgan = Image.open(downloaded_image_srgan)
                    enhancer = ImageEnhance.Sharpness(final_image_srgan)
                    factor = 2
                    final_image_srgan = enhancer.enhance(factor)
                    
                    final_srgan = time.time()
                    execution_time_srgan = final_srgan-start_srgan
                    st.image(final_image_esrgan, caption='This is how your final image from ESRGAN_MODEL looks like 😉')
                    st.write('Execution time : {} second'.format(str(round(execution_time_esrgan,4))))
                    # img_esrgan = PIL.Image.open('downloaded_image_esrgan')
                    # print(brisque.score(img_esrgan))
                    obj = BRISQUE(url=False)
                    img_brisque_esrgan = cv2.imread(downloaded_image_esrgan)

                    score_esrgan = abs(obj.score(img_brisque_esrgan))
                    st.write('Brisque score : ',str(score_esrgan))

                    #st.write( str(brisque.score(img_esrgan)))
                    st.image(final_image_srgan, caption='This is how your final image from SRGAN_MODEL looks like 😉')
                    st.write('Execution time : {} second'.format(str(round(execution_time_srgan,4))))
                    img_brisque_srgan = cv2.imread(downloaded_image_srgan)
                    score_srgan = abs(obj.score(img_brisque_srgan))-score_esrgan
                    st.write('Brisque score : ',str(score_srgan))
                    # render image-comparison
                    #st.set_page_config(page_title="Image-Comparison Example", layout="centered")
                    image_comparison(
                        img1=final_image_esrgan,
                        img2=final_image_srgan,
                        label1='ESRGAN',
                        label2='SRGAN',

                    )
                
            
    else:
        st.warning('⚠ Please upload your Image file 😯')


    st.markdown("<br><hr><center>Made with ❤️ by <a href='mailto:thucdangvan020999@gmail.com?subject=ISR using ESRGAN WebApp!&body=Please specify the issue you are facing with the app.'><strong>ЛЫУ ТИЕН ДАТ 👱‍♂️ КМБО-03-19 </strong></a></center><hr>", unsafe_allow_html=True)


# title = "Face Real ESRGAN UpScale: 2x 4x 8x"
# description = "This is an unofficial demo for Real-ESRGAN. Scales the resolution of a photo. This model shows better results on faces compared to the original version.<br>Telegram BOT: https://t.me/restoration_photo_bot"
# article = "<div style='text-align: center;'>Twitter <a href='https://twitter.com/DoEvent' target='_blank'>Max Skobeev</a> | <a href='https://huggingface.co/sberbank-ai/Real-ESRGAN' target='_blank'>Model card</a>  <center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_face_esrgan' alt='visitor badge'></center></div>"


if choice == 'ISR for Face':
    st.image(main_image,use_column_width='auto')
    st.markdown("<h1 style='text-align: center; color: red;'>👩‍🦳 👩‍🦳 ISR FOR FACE 🧑‍🦳 🧑‍🦳</h1>", unsafe_allow_html=True)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


    add_bg_from_local('static/background1.png')    

    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image 🚀", type=["png","jpg","bmp","jpeg"])
    if uploaded_file is not None:
            with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
                f.write((uploaded_file).getbuffer())
            with st.spinner(f"Working... 💫"):
                uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
                st.image(uploaded_image, caption='This is the original image 😉')
                model = st.radio(
                "Choose resolution model",
                ('  2X  ', '  4X  ', '  8X  '))

                model_face = instantiate_model_face(model)
                downloaded_image_face = os.path.abspath(os.path.join(download_path,str("output_1"+uploaded_file.name)))
                image_super_resolution_face(uploaded_image, downloaded_image_face, model_face)
                final_image_face = Image.open(downloaded_image_face)
                st.image(final_image_face, caption='This is how your final image from REAL-ESRGAN-FACE looks like 😉')



    else:
        st.warning('⚠ Please upload your Image file 😯')


    st.markdown("<br><hr><center>Made with ❤️ by <a href='mailto:thucdangvan020999@gmail.com?subject=ISR using ESRGAN WebApp!&body=Please specify the issue you are facing with the app.'><strong>ЛЫУ ТИЕН ДАТ 👱‍♂️ КМБО-03-19 </strong></a></center><hr>", unsafe_allow_html=True)




if choice == "ISR for ANIME":

    st.image(main_image,use_column_width='auto')
    st.markdown("<h1 style='text-align: center; color: red;'>😹😹😹  ISR FOR ANIME 😹😹😹</h1>", unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


    add_bg_from_local('static/background2.png')    

    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image 🚀", type=["png","jpg","bmp","jpeg"])

    if uploaded_file is not None:
            with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
                f.write((uploaded_file).getbuffer())
            with st.spinner(f"Working... 💫"):
                uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
                st.image(uploaded_image, caption='This is the original image 😉')


                image_ = cv2.imread(uploaded_image)
                image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)

                image_anime = upscale(image_)
                st.image(image_anime, caption='This is how your final image from REAL-ESRGAN-ANIMLE looks like 😉')
                
            
    else:
        st.warning('⚠ Please upload your Image file 😯')


    st.markdown("<br><hr><center>Made with ❤️ by <a href='mailto:thucdangvan020999@gmail.com?subject=ISR using ESRGAN WebApp!&body=Please specify the issue you are facing with the app.'><strong>ЛЫУ ТИЕН ДАТ 👱‍♂️ КМБО-03-19 </strong></a></center><hr>", unsafe_allow_html=True)
