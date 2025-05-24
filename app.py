import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import cv2  # OpenCV for image processing

st.title("Road Lane Line Detection")

uploaded_file = st.file_uploader("Upload a road image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as numpy array (RGB)
    image = mpimg.imread(uploaded_file)

    # Convert RGBA to RGB if needed
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert image from float (0-1) or uint8 (0-255) to uint8 (required by OpenCV)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny edge detection
    edges = cv2.Canny(blur_gray, 50, 150)

    # 4. Create a mask for the region of interest (a polygon covering the lane)
    mask = np.zeros_like(edges)
    ysize = edges.shape[0]
    xsize = edges.shape[1]

    # Define polygon for region of interest
    vertices = np.array([[
        (100, ysize - 1),
        (xsize // 2 - 50, ysize // 2 + 50),
        (xsize // 2 + 50, ysize // 2 + 50),
        (xsize - 100, ysize - 1)
    ]], dtype=np.int32)

    # Fill polygon on mask
    cv2.fillPoly(mask, vertices, 255)

    # Masked edges
    masked_edges = cv2.bitwise_and(edges, mask)

    # Convert edges to a 3-channel image so we can overlay green lines
    edge_color = np.dstack((np.zeros_like(masked_edges), masked_edges, np.zeros_like(masked_edges)))

    # Overlay green edges on the original image
    result = cv2.addWeighted(image_uint8, 1, edge_color, 1, 0)

    # Convert back to RGB for displaying in Streamlit
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Original Image", use_container_width=True)
    st.image(edges, caption="Canny Edges", use_container_width=True)
    st.image(masked_edges, caption="Region of Interest Masked Edges", use_container_width=True)
    st.image(result_rgb, caption="Lane Lines Detected Overlay", use_container_width=True)

else:
    st.write("Please upload an image to start lane line detection.")
