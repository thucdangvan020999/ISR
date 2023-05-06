import cv2
import torch
import numpy as np
import streamlit as st
import RRDBNet_arch as arch
from RealESRGAN import RealESRGAN
from PIL import Image
@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def instantiate_model_face(model_name):
    if model_name:
        device = torch.device('cpu')
        if model_name == "2x":
            model = RealESRGAN(device, scale=2)
            model.load_weights('models/RealESRGAN_x2.pth', download=True)
        elif model_name == "4x":
            model = RealESRGAN(device, scale=4)
            model.load_weights('models/RealESRGAN_x4.pth', download=True)
        else:
            model = RealESRGAN(device, scale=8)
            model.load_weights('models/RealESRGAN_x8.pth', download=True)

        st.write('Model version {}Loaded successfully...'.format(model_name))
        return model
    else:
        st.warning('⚠ Please choose a model !! 😯')


@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def image_super_resolution_face(uploaded_image, downloaded_image, model):
    device = torch.device('cpu')
    img = Image.open(uploaded_image).convert('RGB')
    output = model.predict(img)
    output.save(downloaded_image)
    # cv2.imwrite(downloaded_image, output)
    # blurred = cv2.medianBlur(output, 3)
    # cv2.imwrite(downloaded_image, blurred)



@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def download_success():
    st.balloons()
    st.success('✅ Download Successful !!')
