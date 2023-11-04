import streamlit as st
st.title('CV Group Activity')
import cv2
import numpy as np
import os
from PIL import Image

# Load the image
image = st.file_uploader("", key="image1", type=["png", "jpg", "jpeg"], label_visibility="hidden")

# Check if an image file is uploaded
if image is not None:
    file_details = {"File Name": image.name, "Size": image.size}
    st.write(file_details)
    with open(os.path.join("content", image.name), "wb") as f:
        f.write(image.getbuffer())
        st.title("Original Image")
        st.image(os.path.join("content", image.name))

    # Adjust contrast and brightness
    alpha = 1.5  # Contrast control (1.0 means no change)
    beta = 50  # Brightness control (0 means no change)

    image_data = image.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    st.title("Enhanced Image")
    cv2.imwrite("content/enhanced_image.jpeg", enhanced_image)
    st.image(os.path.join("content", "enhanced_image.jpeg"))

    # Apply Gaussian blur for smoothening
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    st.title("Blurred Image")
    cv2.imwrite("content/blurred_image.jpeg", blurred_image)
    st.image(os.path.join("content", "blurred_image.jpeg"))

    # Apply unsharp masking to sharpen the image
    sharpened_image = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)
    st.title("Sharpened Image")
    cv2.imwrite("content/sharpened_image.jpeg", sharpened_image)
    st.image(os.path.join("content", "sharpened_image.jpeg"))

    # Define a mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(mask, (image.shape[1] // 2, image.shape[0] // 2), 100, (255, 255, 255), -1)

    # Apply masking
    masked_image = cv2.bitwise_and(image, mask)
    st.title("Masked Image")
    cv2.imwrite("content/masked_image.jpeg", masked_image)
    st.image(os.path.join("content", "masked_image.jpeg"))

    # Perform morphological operation (erosion)
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    st.title("Eroded Image")
    cv2.imwrite("content/eroded_image.jpeg", eroded_image)
    st.image(os.path.join("content", "eroded_image.jpeg"))
