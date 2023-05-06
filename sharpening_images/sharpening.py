import cv2
import numpy as np
def sharpening_image(image,model):
    if model == "esrgan":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    else:
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image
