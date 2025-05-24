import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

st.title("Road Lane Line Detection (No OpenCV)")

uploaded_file = st.file_uploader("Upload a road image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = mpimg.imread(uploaded_file)

    ysize = image.shape[0]
    xsize = image.shape[1]

    st.image(image, caption="Original Image", use_column_width=True)

    # Color selection thresholds
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200

    # Create copies for output
    color_select = np.copy(image)
    line_image = np.copy(image)

    # Thresholds mask
    thresholds = (image[:, :, 0] < red_threshold) | \
                 (image[:, :, 1] < green_threshold) | \
                 (image[:, :, 2] < blue_threshold)

    # Define triangular region of interest
    left_bottom = [xsize * 0.1, ysize]
    right_bottom = [xsize * 0.9, ysize]
    apex = [xsize * 0.5, ysize * 0.55]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)

    # Create region mask
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1]))

    # Apply both masks
    color_select[thresholds | ~region_thresholds] = [0, 0, 0]
    line_image[~thresholds & region_thresholds] = [0, 255, 0]  # Green lines

    # Show results
    st.image(color_select, caption="After Color Thresholding", use_column_width=True)
    st.image(line_image, caption="Detected Lane Lines", use_column_width=True)

else:
    st.info("Upload an image to start lane line detection.")

