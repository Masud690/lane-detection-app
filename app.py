import streamlit as st
import numpy as np
import cv2
from PIL import Image

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([
        [(int(0.1*width), height),
         (int(0.45*width), int(0.6*height)),
         (int(0.55*width), int(0.6*height)),
         (int(0.9*width), height)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img

def lane_detection_pipeline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=100)
    line_img = draw_lines(image, lines)
    combo = cv2.addWeighted(image, 0.8, line_img, 1, 1)
    return combo

# Streamlit App
st.set_page_config(page_title="Lane Detection", layout="centered")
st.title("🚗 Lane Line Detection Web App")

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    output = lane_detection_pipeline(image)
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Lane Detection Output", use_column_width=True)
else:
    st.info("Please upload a road image to detect lanes.")

