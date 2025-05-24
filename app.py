import streamlit as st
import numpy as np
import matplotlib.image as mpimg

st.title("Road Lane Line Detection")

uploaded_file = st.file_uploader("Upload a road image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as numpy array
    image = mpimg.imread(uploaded_file)

    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)
    line_image = np.copy(image)

    # Color thresholds (tweak if needed)
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Define region of interest polygon (triangle)
    left_bottom = [100, ysize - 1]
    right_bottom = [xsize - 100, ysize - 1]
    apex = [xsize // 2, ysize // 2]

    # Fit lines for the triangular region
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Color threshold mask
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                       (image[:,:,1] < rgb_threshold[1]) | \
                       (image[:,:,2] < rgb_threshold[2])

    # Region mask
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    # Mask pixels outside thresholds
    color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]

    # Highlight lane lines in bright green
    line_image[~color_thresholds & region_thresholds] = [9, 255, 0]

    # Show images in Streamlit
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(color_select, caption="Color Selection", use_column_width=True)
    st.image(line_image, caption="Lane Lines Detected", use_column_width=True)

else:
    st.write("Please upload an image to start lane line detection.")

