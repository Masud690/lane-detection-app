import streamlit as st
import numpy as np
import matplotlib.image as mpimg

st.title("🚗 Road Lane Line Detection")

uploaded_file = st.file_uploader("Upload a road image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = mpimg.imread(uploaded_file)

    # Copy for processing
    ysize, xsize = image.shape[0], image.shape[1]
    color_select = np.copy(image)
    line_image = np.copy(image)

    # Color threshold to select white/yellow lane lines
    red_thresh = 200
    green_thresh = 200
    blue_thresh = 200

    color_thresholds = (image[:, :, 0] < red_thresh) | \
                       (image[:, :, 1] < green_thresh) | \
                       (image[:, :, 2] < blue_thresh)

    # Triangle region of interest
    left_bottom = [xsize * 0.1, ysize]
    right_bottom = [xsize * 0.9, ysize]
    apex = [xsize * 0.5, ysize * 0.55]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)

    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_mask = (YY > (XX * fit_left[0] + fit_left[1])) & \
                  (YY > (XX * fit_right[0] + fit_right[1]))

    # Apply masks
    color_select[color_thresholds | ~region_mask] = [0, 0, 0]
    line_image[~color_thresholds & region_mask] = [0, 255, 0]  # Green for lanes

    # Show results
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(color_select, caption="After Color Thresholding", use_column_width=True)
    st.image(line_image, caption="Detected Lane Lines", use_column_width=True)

else:
    st.info("Please upload a road image to detect lane lines.")


